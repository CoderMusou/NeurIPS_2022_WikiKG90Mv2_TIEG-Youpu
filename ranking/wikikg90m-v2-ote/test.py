import torch
import os
import sys
model_path = sys.argv[1]
print("start aggregate result")
t_correct_index = []
scores = []
t_pred_top10 = []
for i in range(0, 16):
    try:
        tmp = torch.load(os.path.join(model_path, "{}_{}_0.pkl".format("test", i)))
        scores.append(tmp['h,r->t']['scores'])
    except:
        break

scores = torch.cat(scores, dim=0).numpy()
print(scores.shape)
import numpy as np
os.system('rm %s/test_*_0.pkl' % model_path)
np.save(os.path.join(model_path, "test_candidate_scores.npy"), scores)
print("complete aggregate")
