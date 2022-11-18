from collections import defaultdict, Counter
from multiprocessing import Manager, Process

import numpy as np
from tqdm import tqdm

rule_recall = [
    ('ht.csv', 1.0),
    ('th.csv', 1.0),
    ('ht_ht.csv', 1.0),
    ('th_ht.csv', 1.0),
    ('th_th.csv', 1.0),
    ('rt.csv', 1.0),
    # ('rt_tr_rt.csv', 1.0),
    # ('rh.csv', 1.0),
    # ('rt_hr_rt.csv', 1.0),
    # ('rh_hr_rt.csv', 1.0),
    # ('rh_tr_rt.csv', 1.0)
]

val_rule_recall = [('../candidate_data/rule/valid/' + k, v) for (k, v) in rule_recall]
test_rule_recall = [('../candidate_data/rule/test/' + k, v) for (k, v) in rule_recall]

pie_recall = [
    ('v0.npy', 1.0),
    ('v1.npy', 1.0),
    ('v2.npy', 1.0),
    # ('v3.npy', 1.0),
    # ('v4.npy', 1.0),
]

val_pie_recall = [('../candidate_data/pie/val_t_candidate_' + k, v) for (k, v) in pie_recall]
test_pie_recall = [('../candidate_data/pie/test_candidate_' + k, v) for (k, v) in pie_recall]

val_bert_recall = [
    ('../candidate_data/bert/validation_result2w_relatio_removed.npy', 1.0),
    ('../candidate_data/bert/validation_result2w_removed.npy', 1.0)
]

test_bert_recall = [
    ('../candidate_data/bert/testchallenge_result2w_relation.npy', 1.0),
    ('../candidate_data/bert/testchallenge_result2w.npy', 1.0)
]


val_hr = np.load("../dataset/wikikg90m-v2-pie/processed/val_hr.npy")
val_t = np.load("../dataset/wikikg90m-v2-pie/processed/val_t.npy")
test_hr = np.load("../dataset/wikikg90m-v2-pie/processed/test-challenge_hr.npy")

# get ground truth
ground_truth = defaultdict(list)
for (h, r), t in tqdm(zip(val_hr, val_t)):
    ground_truth[f"{h}_{r}"].append(t)

val_hr2t = defaultdict(lambda: Counter())
test_hr2t = defaultdict(lambda: Counter())
recall_hrt = set()
get_test = True


# process numpy file
def txt_file(file_name, file_weight, hr2t, file_type='valid'):
    recall_num = 0
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(',')
            if line[-1]:
                ts = set([int(x) for x in line[2:]])
                hr2t[f"{line[0]}_{line[1]}"].update(dict.fromkeys(ts, file_weight))
                if file_type == 'valid':
                    for g_t in ground_truth[f"{line[0]}_{line[1]}"]:
                        if g_t in ts:
                            recall_num += 1
                            recall_hrt.add(f"{line[0]}_{line[1]}_{g_t}")
    if file_type == 'valid':
        print("recall num: {}, recall rate: {}.".format(recall_num, recall_num / len(val_t)))
    print()


# process numpy file
def numpy_file(file_name, file_weight, hr2t, h_r, file_type='valid'):
    datas = np.load(file_name)
    if 'bert' in file_name:
        datas = datas[:, :1000]
    print('data shape', datas.shape)
    recall_num = 0
    counted = []
    for ts, (h, r) in tqdm(zip(datas, h_r), total=datas.shape[0]):
        ts = set(ts) - {-1}
        hr2t[f"{h}_{r}"].update(dict.fromkeys(ts, file_weight))
        if file_type == 'valid' and f"{h}_{r}" not in counted:
            counted.append(f"{h}_{r}")
            for g_t in ground_truth[f"{h}_{r}"]:
                if g_t in ts:
                    recall_num += 1
                    recall_hrt.add(f"{h}_{r}_{g_t}")
    if file_type == 'valid':
        print("recall num: {}, recall rate: {}.".format(recall_num, recall_num / len(val_t)))
    print()


# validation bert recall
print("=========  validation bert recall  =========")
for file, weight in val_bert_recall:
    print("process rule file: %s, weight: %.2f" % (file, weight))
    numpy_file(file, weight, val_hr2t, val_hr)

# validation rule recall
print("=========  validation rule recall  =========")
for file, weight in val_rule_recall:
    print("process rule file: %s, weight: %.2f" % (file, weight))
    txt_file(file, weight, val_hr2t)

# validation pie recall
print("=========  validation PIE recall  =========")
for file, weight in val_pie_recall:
    print("process rule file: %s, weight: %.2f" % (file, weight))
    numpy_file(file, weight, val_hr2t, val_hr)


val_res = []
test_res = []
candidate_num = 20000
recall = 0

print("start generate validation candidate ...", end="\n\n")
for (h, r), t in tqdm(zip(val_hr, val_t)):
    val_ts = np.array(list(val_hr2t[f"{h}_{r}"])[:candidate_num])
    val_ts = np.concatenate([val_ts, np.array([-1] * (candidate_num - len(val_ts)))], 0)
    val_res.append(val_ts)
    if t in val_ts:
        recall += 1
new_val_candidate = np.vstack(val_res)
np.save("./valid_candidate_%d_%.4f.npy" % (candidate_num, recall / 15000.0), new_val_candidate.astype('int32'))
print("valid candidate saved!", end="\n\n")
print("recall: %d, recall rate: %.4f" % (recall, recall / 15000.0))
print(len(recall_hrt))


print("start generate test candidate ...", end="\n\n")
if get_test:
    # test bert recall
    print("=========  test bert recall  =========")
    for file, weight in test_bert_recall:
        print("process rule file: %s, weight: %.2f" % (file, weight))
        numpy_file(file, weight, test_hr2t, test_hr, 'test')
    # test rule recall
    print("=========  test rule recall  =========")
    for file, weight in test_rule_recall:
        print("process rule file: %s, weight: %.2f" % (file, weight))
        txt_file(file, weight, test_hr2t, 'test')

    # test pie recall
    print("=========  test PIE recall  =========")
    for file, weight in test_pie_recall:
        print("process rule file: %s, weight: %.2f" % (file, weight))
        numpy_file(file, weight, test_hr2t, test_hr, 'test')

    for (h, r) in tqdm(test_hr):
        test_ts = np.array(list(test_hr2t[f"{h}_{r}"])[:candidate_num])
        test_ts = np.concatenate([test_ts, np.array([-1] * (candidate_num - len(test_ts)))], 0)
        test_res.append(test_ts)
    new_test_candidate = np.vstack(test_res).astype('int32')
    np.save("./test_candidate_%d_%.4f.npy" % (candidate_num, recall / 15000.0), new_test_candidate.astype('int32'))
