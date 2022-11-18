import argparse
import os
import torch
print(torch.__version__)
import numpy as np
import pip
# pip.main(['install', 'faiss-gpu'])
try:
    import faiss
except:
    os.system("/usr/bin/python3 -m pip install --upgrade pip")
    os.system("pip install faiss-gpu==1.7.1")
    # os.system("pip install faiss-gpu")
    import faiss

import time
import pickle as pkl
# the implementation refers to knnlm

parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, default="RotatE_wikikg90m_shallow_d_256_g_10.02/hr_embedding_valid_0.npy", help='memmap where keys and vals are stored')
parser.add_argument('--train_hrt', type=str, default="wikikg90m-v2/processed/train_hrt.npy", help='training set hrts')
parser.add_argument('--test_hr', type=str, default="WikiKG90Mv2/ft_local/wikikg90m-v2/processed/val_hr.npy", help='training set hrts')
parser.add_argument('--fanswer', type=str, default="RotatE_wikikg90m_shallow_d_256_g_10.02/validation_true_tail.npy", help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, default=91230610, help='number of items saved in the datastore memmap')
parser.add_argument('--topk', type=int, default=20000, help='Size of each key')
parser.add_argument('--tails_to_exclude', type=str, default='faiss/val_tail_to_remove.pkl', help='tails to exclude candidates')
parser.add_argument('--faiss_index', type=str, default="faiss_index", help='file to write the faiss index')
parser.add_argument('--fresult', type=str, default="validation_result2w.npy", help='file to write the faiss index')
parser.add_argument('--fresult_dist', type=str, default="validation_result2w_dist.npy", help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=500000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')
parser.add_argument('--load-multiple-files', default=False, action='store_true')
parser.add_argument('--multiple-key-files', type=str, default=None)
parser.add_argument('--multiple-val-files', type=str, default=None)
parser.add_argument('--multiple-files-size', type=str, default=None)
parser.add_argument('--concat-file-path', type=str, default=None)

args = parser.parse_args()

print(args)
training_triples = np.load(args.train_hrt).astype(np.int32)
test_head_rel = np.load(args.test_hr).astype(np.int32)


if not os.path.isfile(args.tails_to_exclude):
    triple_key_dict = dict()
    for i in range(len(training_triples)):
        cur_key = (training_triples[i][0], training_triples[i][1])
        cur_val = training_triples[i][1]
        if cur_key not in triple_key_dict:
            triple_key_dict[cur_key] = set([cur_val])
        else:
            triple_key_dict[cur_key].add(cur_val)
    with open(args.tails_to_exclude, 'wb') as fp:
        pkl.dump(triple_key_dict, fp)
        fp.close()
else:
    triple_key_dict = pkl.load(open(args.tails_to_exclude, 'rb'))
start_time = time.time()
tails_to_exclude = []
for i in range(len(test_head_rel)):
    if i % 100 == 0:
        print("remove tails for {} examples with {} seconds".format(i, time.time()-start_time))
    cur_key = (test_head_rel[i][0],test_head_rel[i][1])
    if cur_key in triple_key_dict:
        tails_to_exclude.append(np.array(list(triple_key_dict[cur_key])))
    else:
        tails_to_exclude.append(np.array([-1]))

print("finish calculating excluding tails...")

query_vecs = np.load(args.dstore_mmap).astype(np.float32)
answers = np.load(args.fanswer) #1500*1
print("answer shape ", answers.shape)

#load faiss index from file
index = faiss.read_index(args.faiss_index)
#search with batchsize
result, distances = [], []

batch_num = len(answers) // 100
for i in range(batch_num):
    cur_distance, cur_indexs = index.search(query_vecs[i*100:(i+1)*100], args.topk)
    result.append(cur_indexs)
    distances.append(cur_distance)

result_arr = np.stack(result, axis=0)
result_arr = result_arr.reshape(-1, args.topk)
distance_arr = np.stack(distances, axis=0) #15000, 20000
distance_arr = distance_arr.reshape(-1, args.topk)
print(result_arr.shape)
recall = 0

result_new = []
distance_new = []

for i in range(len(distance_arr)):
    cur_to_exclude = tails_to_exclude[i]
    exist_judge = np.isin(result_arr[i], cur_to_exclude)
    result_excluded_idxs = np.nonzero(1-exist_judge)
    cur_result = result_arr[result_excluded_idxs]
    cur_distance = distance_arr[result_excluded_idxs]

    if answers[i] in cur_result:
        recall += 1
    result_new.append(cur_result + np.array([-1]*len(args.topk-len(cur_result))))
    distance_new.append(cur_distance + np.array([-1]*len(args.topk-len(cur_result))))

result_new = np.stack(result_new, axis=0).reshape(-1, args.topk)
distance_new = np.stack(distance_new, axis=0).reshape(-1, args.topk)

print("the recall is {}({}/{})".format(recall/(len(distance_arr)+0.0), recall , len(distance_arr)))
with open(args.fresult_dist, 'wb') as fout:
    np.save(fout, distance_new)
    fout.close()
with open(args.fresult, 'wb') as fout:
    np.save(fout, result_new)
    fout.close()