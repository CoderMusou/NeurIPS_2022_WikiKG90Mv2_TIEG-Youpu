import argparse
import logging
import time

import torch

from data_processer import load_data, load_data_wikikg
from train import train, infer, dist_train, dist_infer
import os


def print_setting(args):
    print('')
    print('=============================================')
    for arg in vars(args):
        print('{:20}:{}'.format(arg, getattr(args, arg)))
    print('=============================================')
    print('')


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    log_file = os.path.join(args.save_path, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, help='use gpu', action='store_true')

    parser.add_argument('--dataset', type=str, default='YAGO3-10', help='dataset name')
    parser.add_argument('--steps', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1536, help='batch size')
    parser.add_argument('--cpu_num', type=int, default=16, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id',
                        help='type of relation features: id, bow, bert')

    # settings for model
    parser.add_argument('--add_reverse', type=bool, default=True,
                        help='whether add reverse triples')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8,
                        help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='pna',
                        help='neighbor aggregator: mean, sum, pna')
    parser.add_argument('--neg_sample_num', type=int, default=10,
                        help='number of sampled neighbors for one hop')
    parser.add_argument('--uni_weight', default=False, help='sample weight', action='store_true')
    parser.add_argument('--use_bce', default=False, help='loss type', action='store_true')
    parser.add_argument('--use_ranking_loss', type=bool, default=True, help='Ranking loss')
    parser.add_argument('--gamma', type=float,
                        default=1.0, help='max length of a path')
    parser.add_argument('--margin', type=float,
                        default=3.0, help='max length of a path')

    # settings for wikikg90mv2 dataset
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--nentity', type=int, default=91230610, help='max length of a path')
    parser.add_argument('--nrelation', type=int, default=1387, help='max length of a path')

    # settings for distributed training
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--print_on_screen', action='store_true')
    parser.add_argument('--warm_up', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--up_sampling', action='store_true')
    parser.add_argument('--dist_train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--mp', action='store_true')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--infer_checkpoint', type=str)
    parser.add_argument('--infer_path', type=str)

    args = parser.parse_args()
    datetime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args.save_path = os.path.join(args.data_path, args.dataset, datetime)
    print(args.save_path)
    if args.infer:
        args.save_path = args.infer_path
    if args.save_path and not os.path.exists(args.save_path) and (args.local_rank == 0 or args.local_rank == -1):
        os.makedirs(args.save_path)
    print_setting(args)

    # set_logger(args)

    if args.dataset == 'WikiKG90Mv2':
        data = load_data_wikikg(args, args.nentity, args.nrelation, args.data_path)
    else:
        data = load_data(args)

    if args.infer:
        assert args.infer_checkpoint
        if args.mp:
            torch.multiprocessing.set_start_method('spawn')
            dist_infer(args, data)
        else:
            infer(args, data)
    else:
        if args.dist_train:
            dist_train(args, data)
        elif args.checkpoint:
            train(args, data, args.checkpoint)
        else:
            train(args, data)


if __name__ == '__main__':
    main()

