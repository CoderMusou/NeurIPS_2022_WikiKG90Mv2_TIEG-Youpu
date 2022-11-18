# -*- coding: utf-8 -*-
#
# infer.py
#
# Copyright 2021 PGL, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import argparse

import numpy as np
from tqdm import tqdm
from math import ceil
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from dglke.utils import load_model_config, load_raw_triplet_data, load_triplet_data
from dglke.utils import get_compatible_batch_size
from dglke.models.infer import ScoreInfer
from dglke.dataloader import get_dataset, EvalDataset
from dglke.utils import CommonArgParser
from ogb.lsc import WikiKG90MEvaluator
from dglke.train_pytorch import load_model, load_model_from_checkpoint
from dglke.models.pytorch.tensor_models import thread_wrapped_func


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument(
            '--infer_valid',
            action='store_true',
            help='Evaluate the model on the validation set in the training.')
        self.add_argument(
            '--infer_test',
            action='store_true',
            help='Evaluate the model on the validation set in the training.')
        self.add_argument(
            '--model_path',
            type=str,
            default='ckpts',
            help='the place where to load the model.')


@thread_wrapped_func
def get_embedding(args, model, config, rank, valid_samplers, test_samplers):
    torch.set_num_threads(args.num_thread)
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(
            args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[
            0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()

    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(torch.device('cuda:' + str(gpu_id)))

    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    if args.encoder_model_name in ['roberta', 'concat']:
        model.transform_net = model.transform_net.to(
            torch.device('cuda:' + str(gpu_id)))
    start = time.time()
    with torch.no_grad():
        hr_embeddings = []
        for sampler in valid_samplers:
            for query, ans, candidate in tqdm(
                    sampler,
                    total=ceil(sampler.num_edges / sampler.batch_size)):
                hr_embedding = model.forward_validate_hr_wikikg(query, ans, sampler.mode, gpu_id).squeeze(
                    1).detach().cpu().numpy()
                print(hr_embedding.shape)
                hr_embeddings.append(hr_embedding)
        hr_embeddings = np.concatenate(hr_embeddings, 0)
        print("[{}] finished {} forward".format(rank, "valid"))
        print("cost %s" % (time.time() - start))
        print(hr_embeddings.shape)
        np.save(os.path.join(args.model_path, "hr_embedding_{}.npy".format(0)), hr_embeddings)

        hr_embeddings = []
        for sampler in test_samplers:
            for query, ans, candidate in tqdm(
                    sampler,
                    total=ceil(sampler.num_edges / sampler.batch_size)):
                hr_embedding = model.forward_validate_hr_wikikg(query, ans, sampler.mode, gpu_id).squeeze(
                    1).detach().cpu().numpy()
                print(hr_embedding.shape)
                hr_embeddings.append(hr_embedding)
        hr_embeddings = np.concatenate(hr_embeddings, 0)
        print("[{}] finished {} forward".format(rank, "test"))
        print("cost %s" % (time.time() - start))
        print(hr_embeddings.shape)
        np.save(os.path.join(args.model_path, "hr_embedding_{}.npy".format(1)), hr_embeddings)
    # return input_dict


def use_config_replace_args(args, config):
    for key, value in config.items():
        if key != "data_path":
            setattr(args, key, value)
    return args


def main():
    import os
    import torch
    args = ArgParser().parse_args()
    config = load_model_config(os.path.join(args.model_path, 'config.json'))
    args = use_config_replace_args(args, config)
    args.eval_batch_size = 1
    args.num_proc = 1
    args.eval_percent = 1.0
    dataset = get_dataset(args, args.data_path, args.dataset, args.format,
                          args.delimiter, args.data_files,
                          args.has_edge_importance)
    print("the n_entities:{}, n_relations:{}, entity_feat.shape:{}".format(
        dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1]))
    # model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])
    model = load_model_from_checkpoint(
        args, dataset.n_entities, dataset.n_relations, args.model_path,
        dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])
    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.relation_feat.emb = dataset.relation_feat
    model.evaluator = WikiKG90MEvaluator()
    print("init the model done, the proc_num:{} will load the model".format(
        args.num_proc))
    if args.num_proc > 1 or args.async_update:
        model.share_memory()
    eval_dataset = EvalDataset(dataset, args)

    if args.num_proc >= 1:
        # valid_sampler_heads = []
        valid_sampler_tails = []
        for i in range(args.num_proc):
            print("creating valid sampler for proc %d" % i)
            t1 = time.time()
            valid_sampler_tail = eval_dataset.create_sampler(
                'valid',
                args.batch_size_eval,
                args.neg_sample_size_eval,
                args.neg_sample_size_eval,
                args.eval_filter,
                mode='tail',
                num_workers=args.num_workers,
                rank=i,
                ranks=args.num_proc)
            valid_sampler_tails.append(valid_sampler_tail)
            print("Valid sampler for proc %d created, it takes %s seconds"
                  % (i, time.time() - t1))

        test_challenge_sampler_tails = []
        for i in range(args.num_proc):
            print("creating test_challenge sampler for proc %d" % i)
            t1 = time.time()
            test_challenge_sampler_tail = eval_dataset.create_sampler(
                'test-challenge',
                args.batch_size_eval,
                args.neg_sample_size_eval,
                args.neg_sample_size_eval,
                args.eval_filter,
                mode='tail',
                num_workers=args.num_workers,
                rank=i,
                ranks=args.num_proc)
            test_challenge_sampler_tails.append(test_challenge_sampler_tail)
            print("test-challenge sampler for proc %d created, it takes %s seconds"
                  % (i, time.time() - t1))
    get_embedding(args, model, config, 0, [valid_sampler_tails[0]], [test_challenge_sampler_tails[0]])


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()

