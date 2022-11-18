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
            '--model_path',
            type=str,
            default='ckpts',
            help='the place where to load the model.')


@thread_wrapped_func
def get_embedding(args, model, rank, st, ed, chunk_size, mode="valid"):
    if st*chunk_size >= model.n_entities:
        return
    print("rank %s, st %s, ed %s" % (rank, st*chunk_size, ed*chunk_size))
    torch.set_num_threads(args.num_thread)
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(
            args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[
            0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    print("async complete")
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(torch.device('cuda:' + str(gpu_id)))

    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    if args.encoder_model_name in ['roberta', 'concat']:
        model.transform_net = model.transform_net.to(
            torch.device('cuda:' + str(gpu_id)))
    start = time.time()
    all_candidate = torch.arange(model.n_entities)

    entity_embeddings = []
    print("start")
    with torch.no_grad():
        for c in tqdm(range(st, ed)):
            if c*chunk_size >= model.n_entities:
                break
            embedding = model.forward_entity_embedding(
                all_candidate[c * chunk_size: min(model.n_entities, (c + 1) * chunk_size)].unsqueeze(0), "h,r->t",
                gpu_id).detach().cpu().numpy()
            # print(embedding.shape)
            entity_embeddings.append(embedding)
        entity_embeddings = np.concatenate(entity_embeddings, 0)
        print("[{}] finished {} forward, range {} to {}".format(rank, mode, st*chunk_size, ed*chunk_size))
        print("cost %s" % (time.time() - start))
        np.save(os.path.join(args.model_path, "entity_embedding_{}.npy".format(rank)), entity_embeddings)
        print("save cost %s" % (time.time() - start))
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
    args.num_proc = 4
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
    # model.evaluator = WikiKG90MEvaluator()
    print("init the model done, the proc_num:{} will load the model".format(
        args.num_proc))
    if args.num_proc > 1 or args.async_update:
        model.share_memory()
    # eval_dataset = EvalDataset(dataset, args)

    if args.num_proc >= 1:
        # all_candidate = torch.arange(model.n_entities)
        chunk_size = 100000
        chunks = int(model.n_entities / chunk_size) + 1
        chunk_on_proc = (chunks // args.num_proc) + 1
        process_list = []
        for i in range(args.num_proc):
            proc = mp.Process(target=get_embedding, args=(args, model, i, i*chunk_on_proc, (i+1)*chunk_on_proc, chunk_size, "valid"))
            proc.start()
            process_list.append(proc)
        for proc in process_list:
            proc.join()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()

