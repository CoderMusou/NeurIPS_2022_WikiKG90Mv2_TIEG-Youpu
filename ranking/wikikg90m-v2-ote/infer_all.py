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
from tqdm import tqdm
from math import ceil
from collections import defaultdict
import numpy as np

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
def infer_proc(args, model, config, rank, valid_samplers, test_samplers):
    # torch.set_num_threads(args.num_thread)
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

    with torch.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        scores = defaultdict(list)
        for sampler in valid_samplers:
            for query, ans, candidate in tqdm(
                    sampler,
                    total=ceil(sampler.num_edges / sampler.batch_size)):
                log, score = model.forward_test_wikikg(query, ans, candidate,
                                                       sampler.mode, gpu_id)
                log = log.cpu()
                score = score.cpu()
                logs[sampler.mode].append(log)
                answers[sampler.mode].append(ans)
                scores[sampler.mode].append(score)
        print("[{}] finished {} forward".format(gpu_id, "valid"))
        input_dict = {'h,r->t': {
            't_correct_index': torch.cat(answers['h,r->t'], 0),
            't_pred_top10': torch.cat(logs['h,r->t'], 0)
        }}
        input_dict['h,r->t']['scores'] = torch.cat(scores["h,r->t"], 0)
        torch.save(input_dict,
                   os.path.join(args.model_path, "{}_{}_0.pkl".format("valid",
                                                                      rank)))
        logs = defaultdict(list)
        answers = defaultdict(list)
        scores = defaultdict(list)
        for sampler in test_samplers:
            for query, ans, candidate in tqdm(
                    sampler,
                    total=ceil(sampler.num_edges / sampler.batch_size)):
                log, score = model.forward_test_wikikg(query, ans, candidate,
                                                       sampler.mode, gpu_id)
                log = log.cpu()
                score = score.cpu()
                logs[sampler.mode].append(log)
                answers[sampler.mode].append(ans)
                scores[sampler.mode].append(score)
        print("[{}] finished {} forward".format(gpu_id, "test"))
        input_dict = {'h,r->t': {
            't_pred_top10': torch.cat(logs['h,r->t'], 0)
        }}

        input_dict['h,r->t']['scores'] = torch.cat(scores["h,r->t"], 0)
        torch.save(input_dict,
                   os.path.join(args.model_path, "{}_{}_0.pkl".format("test",
                                                                      rank)))


def use_config_replace_args(args, config):
    for key, value in config.items():
        if key != "data_path":
            setattr(args, key, value)
    return args


def main():
    args = ArgParser().parse_args()
    config = load_model_config(os.path.join(args.model_path, 'config.json'))
    args = use_config_replace_args(args, config)
    args.eval_batch_size = 1
    args.num_proc = 4
    args.eval_percent = 1.0
    dataset = get_dataset(args, args.data_path, args.dataset, args.format,
                          args.delimiter, args.data_files,
                          args.has_edge_importance)
    print("the n_entities:{}, n_relations:{}, entity_feat.shape:{}".format(
        dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1]))
    #model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])
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

        procs = []
        for i in range(0, args.num_proc):
            proc = mp.Process(
                target=infer_proc,
                args=(args, model, config, i, [valid_sampler_tails[i]],
                      [test_challenge_sampler_tails[i]]))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()

