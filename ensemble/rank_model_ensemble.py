import os
import sys

import numpy as np
import itertools as it
import random

valid_candidate_path = sys.argv[1]
valid_correct_t_path = sys.argv[2]
test_candidate_path = sys.argv[3]
output_path = sys.argv[4]

valid_labels = np.load(valid_correct_t_path).astype(int)
candidate_valid = np.load(valid_candidate_path).astype(int)
candidate_test = np.load(test_candidate_path).astype(int)


# MRR calculation
def acc_eval(pred, topk=10):
    top_ind = np.argpartition(pred, -topk)[:, -topk:]
    pred_cur_idxs = np.take_along_axis(top_ind, np.argsort(-np.take_along_axis(pred, top_ind, axis=1), axis=1), axis=1)

    pred_max = np.take_along_axis(candidate_valid, pred_cur_idxs, axis=1)

    tmp_x = np.nonzero(valid_labels.reshape(-1, 1) == pred_max)
    tmp_x0 = np.array(tmp_x[0])
    tmp_x1 = np.array(tmp_x[1])
    tmp = np.stack([tmp_x0, tmp_x1], axis=1)
    rr = np.zeros(len(valid_labels))
    rr[tmp[:, 0]] = 1. / (tmp[:, 1] + 1.)
    return float(rr.mean().item())


def weight_search(cur_model_path, weight_cand, is_random=False):
    all_pred = []
    for ind, path in enumerate(cur_model_path):
        all_pred.append(np.load(path[0], 'r').astype("double"))
    best_mrr = -1
    pre_weight = None
    for i, cur_weight in enumerate(it.product(*weight_cand)):
        if is_random:
            cur_weight = np.asarray([random.random() for _ in range(len(cur_weight))])
        else:
            cur_weight = np.asarray(cur_weight)
        # if sum(cur_weight) < len(cur_weight)-1:
        # if sum(cur_weight) > 1:
        #     continue
        if pre_weight is None:
            # init
            weighted_pred = np.concatenate([np.expand_dims(pred, axis=2) for pred in all_pred], axis=2).dot(cur_weight)
        else:
            delta_weight = cur_weight - pre_weight
            for ind, w in enumerate(delta_weight.tolist()):
                if w != 0:
                    if w == 1:
                        weighted_pred += all_pred[ind]
                    elif w == -1:
                        weighted_pred -= all_pred[ind]
                    else:
                        weighted_pred += w * all_pred[ind]

        ensemble_mrr = acc_eval(weighted_pred)

        if ensemble_mrr > best_mrr:
            best_weight = cur_weight
            best_mrr = ensemble_mrr
        print("loop: {}, current mrr: {}, weight: {}, best mrr: {}, best weight: {}".
              format(i, ensemble_mrr, cur_weight, best_mrr, best_weight.tolist()), flush=True)
        pre_weight = cur_weight

    return best_weight, best_mrr


def ensemble(paths, best_weight, f, candidate, topk=10):
    all_pred = []
    for path in paths:
        all_pred.append(np.load(path, 'r'))

    all_pred = np.concatenate([np.expand_dims(pred, axis=2) for pred in all_pred], axis=2)

    weighted_pred = all_pred.dot(best_weight)

    top_ind = np.argpartition(weighted_pred, -topk)[:, -topk:]
    pred_cur_idxs = np.take_along_axis(top_ind, np.argsort(-np.take_along_axis(weighted_pred, top_ind, axis=1), axis=1),
                                       axis=1)

    pred_max = np.take_along_axis(candidate, pred_cur_idxs, axis=1)

    np.save(f, pred_max)


if __name__ == '__main__':
    print("begin to ensemble!!!")


    cur_model_path = [
        (
            "model1/validation_candidate_scores.npy",
            "model1/test_candidate_scores.npy"
        ),
        (
            "model2/validation_candidate_scores.npy",
            "model2/test_candidate_scores.npy"
        ),
        # ...
        # more models prediction result...
    ]

    weight_search_space = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

    best_weight, best_mrr = weight_search(cur_model_path, weight_search_space, is_random=False)

    print("begin to ensemble dev set")
    os.system("mkdir {}/{}".format(output_path, best_mrr))
    save_dev_path = "{}/{}/dev_ensenble.npy".format(output_path, best_mrr)
    save_test_path = "{}/{}/test_ensenble.npy".format(output_path, best_mrr)

    ensemble([path[0] for path in cur_model_path], best_weight, save_dev_path, candidate_valid, topk=10)
    # reload and evaluation
    dev_result = np.load(save_dev_path, 'r')
    tmp_x = np.nonzero(valid_labels.reshape(-1, 1) == dev_result)
    tmp_x0 = np.array(tmp_x[0])
    tmp_x1 = np.array(tmp_x[1])
    tmp = np.stack([tmp_x0, tmp_x1], axis=1)
    rr = np.zeros(len(valid_labels))
    rr[tmp[:, 0]] = 1. / (tmp[:, 1] + 1.)
    dev_mrr = float(rr.mean().item())
    print("dev_mrr: {}".format(dev_mrr))

    assert dev_mrr == best_mrr, (dev_mrr, best_mrr)

    print("begin to ensemble test set")
    ensemble([path[1] for path in cur_model_path], best_weight, save_test_path, candidate_test)
    test_ensenble = np.load(save_test_path, 'r')
    print("test_ensenble: {}".format(test_ensenble.shape))
    print("test_ensenble: {}".format(test_ensenble[:5, :]))
