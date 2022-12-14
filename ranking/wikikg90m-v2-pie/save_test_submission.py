import os
import pickle
import json
import numpy as np
import sys
from ogb.lsc import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
import pdb
from collections import defaultdict
import torch.nn.functional as F
import torch

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# python save_test_submission.py $SAVE_PATH $NUM_PROC $MODE
if __name__ == '__main__':
    path = sys.argv[1]
    num_proc = int(sys.argv[2])

    with_test_dev = str2bool(sys.argv[3])
    with_test_challenge = str2bool(sys.argv[4])

    valid_candidate_path = sys.argv[5]
    test_dev_candidate_path = sys.argv[6]
    test_challenge_candidate_path = sys.argv[7]

    valid_candidates = np.load(valid_candidate_path)
    test_dev_candidates = np.load(test_dev_candidate_path)
    test_challenge_candidates = np.load(test_challenge_candidate_path)

    all_file_names = os.listdir(path)
    test_file_names = [name for name in all_file_names if '.pkl' in name and 'test' in name]
    valid_file_names = [name for name in all_file_names if '.pkl' in name and 'valid' in name]
    steps = [int(name.split('.')[0].split('_')[-1])
             for name in valid_file_names if 'valid_0' in name]
    steps.sort()
    evaluator = WikiKG90Mv2Evaluator()
    device = torch.device('cpu')

    all_test_dev_dicts = []
    all_test_challenge_dicts = []
    best_valid_mrr = -1
    best_valid_idx = -1
    
    for i, step in enumerate(steps):
        valid_result_dict = defaultdict(lambda: defaultdict(list))
        test_dev_result_dict = defaultdict(lambda: defaultdict(list))
        test_challenge_result_dict = defaultdict(lambda: defaultdict(list))

        for proc in range(num_proc):
            valid_result_dict_proc = torch.load(os.path.join(
                path, "valid_{}_{}.pkl".format(proc, step)), map_location=device)
            for result_dict_proc, result_dict in zip([valid_result_dict_proc], [valid_result_dict]):
                for key in result_dict_proc['h,r->t']:
                    result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())
            if with_test_dev:
                test_result_dict_proc = torch.load(os.path.join(
                    path, "test_dev_{}_{}.pkl".format(proc, step)), map_location=device)
                for result_dict_proc, result_dict in zip([test_result_dict_proc], [test_dev_result_dict]):
                    for key in result_dict_proc['h,r->t']:
                        result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())
            if with_test_challenge:
                test_result_dict_proc = torch.load(os.path.join(
                    path, "test_challenge_{}_{}.pkl".format(proc, step)), map_location=device)
                for result_dict_proc, result_dict in zip([test_result_dict_proc], [test_challenge_result_dict]):
                    for key in result_dict_proc['h,r->t']:
                        result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())

        for result_dict in [valid_result_dict]:
            for key in result_dict['h,r->t']:
                if key == 't_pred_top10':
                    index = np.concatenate(result_dict['h,r->t'][key], 0)
                    temp = []
                    for ii in range(index.shape[0]):
                        temp.append(valid_candidates[ii][index[ii]])
                    result_dict['h,r->t'][key] = np.concatenate(np.expand_dims(temp, 0))
                else:
                    result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0)
        if with_test_dev:
            for result_dict in [test_dev_result_dict]:
                for key in result_dict['h,r->t']:
                    if key == 't_pred_top10':
                        index = np.concatenate(result_dict['h,r->t'][key], 0)
                        temp = []
                        for ii in range(index.shape[0]):
                            temp.append(test_dev_candidates[ii][index[ii]])
                        result_dict['h,r->t'][key] = np.concatenate(np.expand_dims(temp, 0))
                    else:
                        result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0)
        if with_test_challenge:
            for result_dict in [test_challenge_result_dict]:
                for key in result_dict['h,r->t']:
                    if key == 't_pred_top10':
                        index = np.concatenate(result_dict['h,r->t'][key], 0)
                        temp = []
                        for ii in range(index.shape[0]):
                            temp.append(test_challenge_candidates[ii][index[ii]])
                        result_dict['h,r->t'][key] = np.concatenate(np.expand_dims(temp, 0))
                    else:
                        result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0)

        if with_test_dev:
            all_test_dev_dicts.append(test_dev_result_dict)
        if with_test_challenge:
            all_test_challenge_dicts.append(test_challenge_result_dict)

        metrics = evaluator.eval(valid_result_dict)
        metric = 'mrr'
        print("valid-{} at step {}: {}".format(metric, step, metrics[metric]))
        if metrics[metric] > best_valid_mrr:
            best_valid_mrr = metrics[metric]
            best_valid_idx = i

    if with_test_dev:
        best_test_dev_dict = all_test_dev_dicts[best_valid_idx]
        t_pred_top10 = best_test_dev_dict['h,r->t']['t_pred_top10']

        for i in range(len(t_pred_top10)):
            if len(set(t_pred_top10[i])) != len(t_pred_top10[i]):
                pred = np.delete(t_pred_top10[i], np.where(t_pred_top10[i] == -1))
                random_pad = np.random.randint(0, 91230609, size=(10 - len(pred),))
                t_pred_top10[i] = np.concatenate([pred, random_pad])

        evaluator.save_test_submission(best_test_dev_dict, path, 'test-dev')
    if with_test_challenge:
        best_test_challenge_dict = all_test_challenge_dicts[best_valid_idx]
        t_pred_top10 = best_test_challenge_dict['h,r->t']['t_pred_top10']

        for i in range(len(t_pred_top10)):
            if len(set(t_pred_top10[i])) != len(t_pred_top10[i]):
                pred = np.delete(t_pred_top10[i], np.where(t_pred_top10[i] == -1))
                random_pad = np.random.randint(0, 91230609, size=(10 - len(pred),))
                t_pred_top10[i] = np.concatenate([pred, random_pad])

        evaluator.save_test_submission(best_test_challenge_dict, path, 'test-challenge')

