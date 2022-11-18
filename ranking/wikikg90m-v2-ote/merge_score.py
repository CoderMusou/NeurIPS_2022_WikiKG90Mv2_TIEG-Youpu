# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import pickle
import json
import numpy as np
import sys
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
import pdb
from collections import defaultdict
import torch.nn.functional as F
import torch


def get_mrr(step, path, num_proc):
    valid_result_dict = defaultdict(lambda: defaultdict(list))
    for proc in range(num_proc):
        valid_result_dict_proc = torch.load(
            os.path.join(path, "valid_{}_{}.pkl".format(proc, step)),
            map_location=device)
        for result_dict_proc, result_dict in zip([valid_result_dict_proc],
                                                 [valid_result_dict]):
            for key in result_dict_proc['h,r->t']:
                result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][
                    key].numpy())
    for result_dict in [valid_result_dict]:
        for key in result_dict['h,r->t']:
            result_dict['h,r->t'][key] = np.concatenate(
                result_dict['h,r->t'][key], 0)

    metrics = evaluator.eval(valid_result_dict)
    metric = 'mrr'
    return metrics[metric]


# python save_test_submission.py $SAVE_PATH $NUM_PROC
if __name__ == '__main__':
    path = sys.argv[1]
    num_proc = int(sys.argv[2])
    mode = sys.argv[3]

    all_file_names = os.listdir(path)
    valid_file_names = [
        name for name in all_file_names if '.pkl' in name and 'valid' in name
    ]
    steps = [0]
    evaluator = WikiKG90MEvaluator()
    device = torch.device('cpu')

    print(valid_file_names)

    best_valid_mrr = -1
    best_valid_idx = -1

    for i, step in enumerate(steps):
        valid_result_dict = defaultdict(lambda: defaultdict(list))
        for proc in range(num_proc):
            valid_result_dict_proc = torch.load(
                os.path.join(path, "{}_{}_{}.pkl".format(mode, proc, step)),
                map_location=device)
            for result_dict_proc, result_dict in zip([valid_result_dict_proc],
                                                     [valid_result_dict]):
                for key in result_dict_proc['h,r->t']:
                    result_dict['h,r->t'][key].append(result_dict_proc[
                        'h,r->t'][key].numpy())
        for result_dict in [valid_result_dict]:
            for key in result_dict['h,r->t']:
                result_dict['h,r->t'][key] = np.concatenate(
                    result_dict['h,r->t'][key], 0)

        np.save(path + "/{}_scores.npy".format(mode),
                valid_result_dict['h,r->t']['scores'].astype(np.float16))
        if mode == "valid":
            metrics = evaluator.eval(valid_result_dict)
            metric = 'mrr'
            print("valid-{} at step {}: {}".format(metric, step, metrics[
                metric]))
