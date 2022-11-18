import os
import gc
import time
import sys
import math
import traceback

import torch
import numpy as np
from tqdm import tqdm
from KGDataset import get_dataset
from sampler import ConstructGraph, EvalDataset
from utils import CommonArgParser, thread_wrapped_func
import torch.multiprocessing as mp


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'
                          'The positive score will be adjusted '
                          'as pos_score = pos_score * edge_importance')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--test_dev', action='store_true',
                          help='Evaluate the model on the test_dev set in the training.')
        self.add_argument('--test_challenge', action='store_true',
                          help='Evaluate the model on the test_challenge set in the training.')
        self.add_argument('--num_hops', type=int, default=2, help='.')
        self.add_argument('--expand_factors', type=int, default=1000000, help='.')
        self.add_argument('--num_workers', type=int, default=16, help='.')
        self.add_argument('--print_on_screen', action='store_true', help='')
        self.add_argument('--num_candidates', type=int, default=20000, help='')
        self.add_argument('--save_file', type=str, default="test_tail_candidate", help='')
        self.add_argument('--e2r_score_file', type=str, default="e2r_score file", help='')


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


def infer(args, samplers, save_paths, i=0):
    candidates, spos = [], []
    n_hop_find, find, total = 0, 0, 0
    try:
        sampler_id = 0
        spo_id = 0
        for sampler in samplers:
            sampler_id += 1
            for candidate, is_n_hop_find, is_find, spo in tqdm(sampler, disable=not args.print_on_screen, total=sampler.num_edges):
                spo_id += 1
                print("%s: %d-th process, %d-th sampler, %d-th spo" % (time.strftime("%H:%M:%S", time.localtime()),
                                                                       i, sampler_id, spo_id))
                candidates.append(candidate.unsqueeze(0))
                spos.append(spo)
                if is_find == 1:
                    find += 1
                if is_n_hop_find == 1:
                    n_hop_find += 1
                total += 1
                if total % 100 == 0:
                    print("%d-th process hit: %d/%d=%f, n_hop hit: %d/%d=%f" %
                          (i, find, total, float(find)/float(total), n_hop_find, total, float(n_hop_find)/float(total)))
    except Exception as e:
        err = traceback.format_exc()
        f = open(args.data_path + "error.txt", 'a')
        print(f"Process {i} is down!", file=f)
        print(err, file=f)

    candidates = torch.cat(candidates, axis=0)
    spos = torch.cat(spos, axis=0)
    ret = torch.cat([spos, candidates], axis=1).numpy()
    return np.save(save_paths[0], ret)


@thread_wrapped_func
def infer_mp(args, samplers, save_paths, i, rank=0, mode='Test'):
    if args.num_proc > 1:
        torch.set_num_threads(args.num_thread)
    print(f"The {i}-th process stated! The sample num is {len(samplers)}")
    infer(args, samplers, save_paths, i)

def multi_process_get_candidate(args, eval_type, eval_dataset, in_degree):
    valid_samplers, save_paths = [], []
    for i in range(args.num_proc):
        valid_sampler = eval_dataset.create_sampler(eval_type, args.batch_size_eval,
                                                    args.num_hops,
                                                    args.expand_factors,
                                                    'tail-batch',
                                                    in_degree,
                                                    num_workers=args.num_workers,
                                                    num_candidates=args.num_candidates,
                                                    rank=i, ranks=args.num_proc)

        save_file = args.save_file + '_' + \
            str(args.num_candidates) + '_' + str(i) + '.npy'
        if args.dataset == 'wikikg90M':
            save_path = os.path.join(args.save_path, save_file)
        else:
            save_path = os.path.join(args.save_path, args.dataset, save_file)
        save_paths.append(save_path)
        valid_samplers.append(valid_sampler)
    print(save_paths)
    procs = []
    for i in range(args.num_proc):
        proc = mp.Process(target=infer_mp, args=(
            args, [valid_samplers[i]], [save_paths[i]], i))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


def main():
    args = ArgParser().parse_args()
    prepare_save_path(args)
    dataset = get_dataset(args.data_path, args.dataset, args.format,
                          args.delimiter, e2r_score_file=args.e2r_score_file)
    g, in_degree, out_degree = ConstructGraph(dataset, args)
    if args.valid or args.test_dev or args.test_challenge:
        eval_dataset = EvalDataset(g, dataset, args)
    if args.num_proc > 1:
        if args.valid:
            multi_process_get_candidate(args, 'valid', eval_dataset, in_degree)
        elif args.test_dev:
            multi_process_get_candidate(args, 'test_dev', eval_dataset, in_degree)
        elif args.test_challenge:
            multi_process_get_candidate(args, 'test_challenge', eval_dataset, in_degree)
    else:
        if args.valid:
            valid_sampler = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                        args.num_hops,
                                                        args.expand_factors,
                                                        'tail-batch',
                                                        in_degree,
                                                        num_workers=args.num_workers,
                                                        num_candidates=args.num_candidates)
            save_file = args.save_file + '_' + str(args.num_candidates) + '.npy'
            if args.dataset == 'wikikg90M':
                save_path = os.path.join(args.data_path, 'wikikg90m-v2-pie/processed/', save_file)
            else:
                save_path = os.path.join(args.data_path, args.dataset, save_file)
            candidates, spos = infer(args, [valid_sampler], [save_path])
            np.save(save_path, candidates.numpy())


if __name__ == '__main__':
    main()

