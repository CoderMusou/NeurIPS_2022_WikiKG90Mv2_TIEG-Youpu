import sys
from ogb.lsc import WikiKG90Mv2Evaluator
import numpy as np

evaluator = WikiKG90Mv2Evaluator()

t_pred_top10_file_path = sys.argv[1]

t_pred_top10 = np.load(t_pred_top10_file_path)

for i in range(len(t_pred_top10)):
    if len(set(t_pred_top10[i])) != len(t_pred_top10[i]):
        pred = np.delete(t_pred_top10[i], np.where(t_pred_top10[i] == -1))
        random_pad = np.random.randint(0, 91230609, size=(10 - len(pred),))
        t_pred_top10[i] = np.concatenate([pred, random_pad])

input_dict = {}
input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
evaluator.save_test_submission(input_dict=input_dict, dir_path='./', mode='test-challenge')
