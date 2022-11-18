import logging
import os
import json
import time
import traceback
from _thread import start_new_thread
from functools import wraps
from multiprocessing import Queue

import torch
import torch.multiprocessing as mp
import numpy as np
import scipy
import gc
from collections import defaultdict

# from torch.optim.lr_scheduler import LambdaLR
# from warmup_scheduler import GradualWarmupScheduler

from model import PathCon
from utils import sparse_to_tuple
from data_loader import TrainTestDataset, OneShotIterator
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
# from model import train_step, test_step_score, test_step_loss
from sklearn.metrics import precision_recall_curve, roc_auc_score
from scipy.sparse import coo_matrix, vstack


def dist_train(args, data):
    # init
    dist.init_process_group(backend="nccl")
    # Configure gpu for each process
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # extract data
    triplets, n_relations, neighbor_data = data
    train_triplets, valid_triplets, test_triplets, infer_triplets = triplets
    total_num = train_triplets.shape[0]
    train_data = TrainTestDataset(
        train_triplets, n_relations, neighbor_data[0], neighbor_data[1],
        neighbor_data[2], neighbor_data[3], args.context_hops)

    print("number of triplets in train %d, steps in one epoch %d" %
          (train_triplets.shape[0], train_triplets.shape[0] / args.batch_size))
    print("number of relations %d" % n_relations)
    del train_triplets

    # DistributedSampler
    train_sampler = DistributedSampler(train_data)

    if valid_triplets is not None:
        valid_data = TrainTestDataset(
            valid_triplets, n_relations, neighbor_data[0], neighbor_data[1],
            neighbor_data[2], neighbor_data[3], args.context_hops)
    else:
        valid_data = None

    if test_triplets is not None:
        test_data = TrainTestDataset(
            test_triplets, n_relations, neighbor_data[0], neighbor_data[1],
            neighbor_data[2], neighbor_data[3], args.context_hops)
    else:
        test_data = None

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainTestDataset.collate_fn,
        sampler=train_sampler
    )
    train_iterator = OneShotIterator(train_dataloader)

    del neighbor_data

    # define the model
    model = PathCon(args, n_relations)
    # Move the model to the specific gpu
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2,
    )
    # LambdaLR(optimizer, lambda step:  1 / (1 + 1e-5 * step))
    # scheduler_lr = LambdaLR(optimizer, lambda step: 1 / (1 + 1e-6 * step))
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_up,
    #                                           after_scheduler=scheduler_lr)
    optimizer.zero_grad()
    optimizer.step()

    if args.cuda:
        model = model.cuda()

    final_res = None  # acc, mrr, mr, hit1, hit3, hit5
    print('start training ...')

    best_valid_mrr = 0.0
    early_stop = 0
    losses = []

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    for step in range(args.steps):
        # scheduler_warmup.step(step)
        cur_epoch = step // int(total_num / args.batch_size)
        train_sampler.set_epoch(cur_epoch)
        start1 = time.time()
        train_entity_pairs, train_labels, label_sets, tail_edges_lists, tail_masks, tail2relation = next(
            train_iterator)
        sample_time += time.time() - start1

        loss, forward_time_, backward_time_, update_time_ = model.module.train_step(
            optimizer,
            get_feed_dict(train_labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda),
            args.uni_weight)
        forward_time += forward_time_
        backward_time += backward_time_
        update_time += update_time_

        losses.append(loss)

        if len(losses) % 2000 == 0:
            print("[proc %2d]epoch: %2d: step: %4d: train_loss: %4f: lr: %6f" % (local_rank,
                                                                                 cur_epoch,
                                                                                 step,
                                                                                 np.array(losses).mean(),
                                                                                 optimizer.param_groups[0]['lr']))
            losses = []
            gc.collect()
        if (step + 1) % 20000 == 0 and local_rank == 0:
            print('[proc {}][Train] 20000 steps take {:.3f} seconds'.format(local_rank, time.time() - start))
            print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                local_rank, sample_time, forward_time, backward_time, update_time))
            sample_time = 0
            update_time = 0
            forward_time = 0
            backward_time = 0
            start = time.time()

            if valid_data is not None:
                valid_loss = evaluate_loss(valid_data, model, args)
                print("valid_loss: %4f" % (valid_loss))

            if test_data is not None:
                test_loss = evaluate_loss(test_data, model, args)
                print("test_loss: %4f" % (test_data))

            ret, auc = evaluate(valid_data, model, args)
            print("[proc %2d]Valid-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f; AUC: %4f" %
                  (local_rank, ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))

            if ret[0] > best_valid_mrr:
                best_valid_mrr = ret[0]
                early_stop = 0
                print("saving model ...")
                save_model(model.module, args)
            else:
                early_stop += 1
                if early_stop > 50:
                    print("early stop!")
                    break

            if test_data is not None:
                ret, aus = evaluate(test_data, model, args)
                print("Test-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f;  AUC:%4f" %
                      (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))
        else:
            torch.distributed.barrier()

    valid_loss = evaluate_loss(valid_data, model, args)
    print("valid_loss: %4f" % (valid_loss))

    ret, auc = evaluate(valid_data, model, args)
    print("Valid-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f; AUC: %4f" %
          (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))

    if test_data is not None:
        test_loss = evaluate_loss(test_data, model, args)
        print("test_loss: %4f" % (valid_loss))

        ret, auc = evaluate(test_data, model, args)
        print("Test-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f; AUC: %4f" %
              (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))


def train(args, data, checkpoint=None):
    # extract data
    triplets, n_relations, neighbor_data = data
    train_triplets, valid_triplets, test_triplets, infer_triplets = triplets
    train_data = TrainTestDataset(
        train_triplets, n_relations, neighbor_data[0], neighbor_data[1],
        neighbor_data[2], neighbor_data[3], args.context_hops)

    print("number of triplets in train %d, steps in one epoch %d" %
          (train_triplets.shape[0], train_triplets.shape[0] / args.batch_size))
    print("number of relations %d" % n_relations)

    if valid_triplets is not None:
        valid_data = TrainTestDataset(
            valid_triplets, n_relations, neighbor_data[0], neighbor_data[1],
            neighbor_data[2], neighbor_data[3], args.context_hops)
    else:
        valid_data = None

    if test_triplets is not None:
        test_data = TrainTestDataset(
            test_triplets, n_relations, neighbor_data[0], neighbor_data[1],
            neighbor_data[2], neighbor_data[3], args.context_hops)
    else:
        test_data = None

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainTestDataset.collate_fn
    )
    train_iterator = OneShotIterator(train_dataloader)

    del neighbor_data

    if args.up_sampling:
        print("using up sampling ...")
        up_sampling_weight = torch.from_numpy(np.load("../up_sampling_weight.npy")).cuda()
    else:
        up_sampling_weight = None

    # define the model
    model = PathCon(args, n_relations)

    if checkpoint:
        print('Loading checkpoint %s...' % checkpoint)
        args.save_path = checkpoint
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()

    final_res = None  # acc, mrr, mr, hit1, hit3, hit5
    print('start training ...')

    best_valid_mrr = 0.0
    early_stop = 0
    losses = []
    for step in range(args.steps):
        train_entity_pairs, train_labels, label_sets, tail_edges_lists, tail_masks, tail2relation = next(
            train_iterator)

        loss, forward_time, backward_time, update_time = model.train_step(
            optimizer,
            get_feed_dict(
                train_labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda),
            args.uni_weight)

        losses.append(loss)

        if len(losses) % 2000 == 0:
            print("step: %4d: train_loss: %4f" % (step, np.array(losses).mean()))
            losses = []
            gc.collect()
        if step % 20000 == 0:
            if valid_data is not None:
                valid_loss = evaluate_loss(valid_data, model, args)
                print("valid_loss: %4f" % (valid_loss))

            if test_data is not None:
                test_loss = evaluate_loss(test_data, model, args)
                print("test_loss: %4f" % (test_loss))

            ret, auc = evaluate(valid_data, model, args)
            print("Valid-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f; AUC: %4f" %
                  (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))

            if ret[0] > best_valid_mrr:
                best_valid_mrr = ret[0]
                early_stop = 0
                print("saving model ...")
                save_model(model, args)
            else:
                early_stop += 1
                if early_stop > 50:
                    break

            if test_data is not None:
                ret, aus = evaluate(test_data, model, args)
                print("Test-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f;  AUC:%4f" %
                      (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))

    valid_loss = evaluate_loss(valid_data, model, args)
    print("valid_loss: %4f" % (valid_loss))

    ret, auc = evaluate(valid_data, model, args)
    print("Valid-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f; AUC: %4f" %
          (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))

    if test_data is not None:
        test_loss = evaluate_loss(test_data, model, args)
        print("test_loss: %4f" % (valid_loss))

        ret, auc = evaluate(test_data, model, args)
        print("Test-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f; Hit@10: %4f; AUC: %4f" %
              (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))


def infer(args, data):
    # extract data
    triplets, n_relations, neighbor_data = data
    train_triplets, valid_triplets, test_triplets, infer_triplets = triplets

    if valid_triplets is not None:
        valid_data = TrainTestDataset(
            valid_triplets, n_relations, neighbor_data[0], neighbor_data[1],
            neighbor_data[2], neighbor_data[3], args.context_hops)
    else:
        valid_data = None

    model = PathCon(args, n_relations)

    print('Loading checkpoint %s...' % args.infer_checkpoint)
    checkpoint = torch.load(os.path.join(args.infer_checkpoint, 'checkpoint'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.cuda:
        model = model.cuda()

    ret, auc = evaluate(valid_data, model, args)
    print("Valid-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f, Hit@10: %4f; AUC: %4f" %
          (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))
    del data, train_triplets
    # split infer dataset
    chunk_size = 5000 * args.test_batch_size
    chunks = int(len(infer_triplets) / chunk_size) + 1
    for part in range(chunks):
        print("%d/%d" % (part, chunks))
        infer_data = TrainTestDataset(
            infer_triplets[part * chunk_size: (part + 1) * chunk_size], n_relations,
            neighbor_data[0], neighbor_data[1], neighbor_data[2],
            neighbor_data[3], args.context_hops)
        save_prediction(infer_data, model, args, part)


def dist_infer(args, data):
    # extract data
    triplets, n_relations, neighbor_data = data
    train_triplets, valid_triplets, test_triplets, infer_triplets = triplets

    if valid_triplets is not None:
        valid_data = TrainTestDataset(
            valid_triplets, n_relations, neighbor_data[0], neighbor_data[1],
            neighbor_data[2], neighbor_data[3], args.context_hops)
    else:
        valid_data = None

    model = PathCon(args, n_relations)

    print('Loading checkpoint %s...' % args.infer_checkpoint)
    checkpoint = torch.load(os.path.join(args.infer_checkpoint, 'checkpoint'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.cuda:
        model = model.cuda()

    ret, auc = evaluate(valid_data, model, args)
    print("Valid-MRR: %4f; MR: %4f; Hit@1: %4f; Hit@3: %4f; Hit@5: %4f, Hit@10: %4f; AUC: %4f" %
          (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], auc))
    del data, train_triplets
    # split infer dataset
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # chunks = 3
    # num_proc = 4
    # chunk_size = len(infer_triplets) // (chunks * num_proc)
    # for i in range(chunks):
    #     procs = []
    #     for j in range(num_proc):
    #         part = i * chunks + j
    #         print("%d/%d" % (part, (chunks * num_proc)))
    #         start = part * chunk_size
    #         end = (part + 1) * chunk_size if part < (chunks * num_proc) - 1 else len(infer_triplets)
    #         infer_data = TrainTestDataset(
    #             infer_triplets[start: end], n_relations,
    #             neighbor_data[0], neighbor_data[1], neighbor_data[2],
    #             neighbor_data[3], args.context_hops)
    #         # save_prediction(infer_data, model, args, part)
    #         proc = mp.Process(target=save_prediction, args=(infer_data, model, args, j, part))
    #         procs.append(proc)
    #         proc.start()
    #     for proc in procs:
    #         proc.join()

    num_proc = 4
    chunk_size = len(infer_triplets) // num_proc
    procs = []
    for part in range(num_proc):
        print("%d/%d" % (part, num_proc))
        start = part * chunk_size
        end = (part + 1) * chunk_size if part < num_proc - 1 else len(infer_triplets)
        infer_data = TrainTestDataset(
            infer_triplets[start: end], n_relations,
            neighbor_data[0], neighbor_data[1], neighbor_data[2],
            neighbor_data[3], args.context_hops)
        dataloader = DataLoader(
            infer_data,
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=infer_data.collate_fn
        )
        # save_prediction(infer_data, model, args, part)
        proc = mp.Process(target=save_prediction_mp, args=(dataloader, model, args, part, part))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()


def get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, use_cuda):
    feed_dict = {}
    feed_dict["labels"] = labels.cuda() if use_cuda else labels
    feed_dict["label_sets"] = label_sets.cuda() if use_cuda else label_sets
    feed_dict["tail_edges_lists"] = [t.cuda()
                                     for t in tail_edges_lists] if use_cuda else tail_edges_lists
    feed_dict["tail2relation"] = [t.cuda() for t in tail2relation] if use_cuda else tail2relation
    if tail_masks is None:
        tail_masks = [torch.ones(t.size(), dtype=torch.bool) for t in tail2relation]
        feed_dict["tail_masks"] = [t.cuda() for t in tail_masks] if use_cuda else tail_masks
    else:
        feed_dict["tail_masks"] = [t.cuda() for t in tail_masks] if use_cuda else tail_masks

    return feed_dict


def evaluate_loss(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=dataset.collate_fn
    )
    losses = []
    for entity_pairs, labels, label_sets, tail_edges_lists, tail_masks, tail2relation in dataloader:
        if args.local_rank == -1:
            loss = model.test_step_loss(
                get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda)
            )
        else:
            loss = model.module.test_step_loss(
                get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda)
            )
        losses.append(loss)
    return float(np.mean(losses))


def evaluate(dataset, model, args):
    scores_tail_list = []
    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=dataset.collate_fn
    )

    for entity_pairs, labels, label_sets, tail_edges_lists, tail_masks, tail2relation in dataloader:
        if args.local_rank == -1:
            neg_score = model.test_step_score(
                get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda)
            )
        else:
            neg_score = model.module.test_step_score(
                get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda)
            )

        scores_tail_list.append(neg_score)

    scores_tail = torch.cat(scores_tail_list, dim=0)
    ret_tail = calculate_ranking_metrics(
        dataset.triples, scores_tail, dataset.entity2relation)
    return ret_tail, 0.0


def calculate_ranking_metrics(triples, scores_tail, true_relations):
    relations = []
    for idx, triple in enumerate(triples):
        # import pdb; pdb.set_trace()
        head, relation, tail = triple
        relations.append(relation)
        for j in set(true_relations[tail]) - {relation}:
            scores_tail[idx, j] -= float('inf')
    sorted_indices = np.argsort(-scores_tail.cpu().numpy(), axis=1)
    relations = np.array(relations)
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr_tail = float(np.mean(1 / rankings))
    mr_tail = float(np.mean(rankings))
    hit1_tail = float(np.mean(rankings <= 1))
    hit3_tail = float(np.mean(rankings <= 3))
    hit5_tail = float(np.mean(rankings <= 5))
    hit10_tail = float(np.mean(rankings <= 10))

    return (mrr_tail, mr_tail, hit1_tail, hit3_tail, hit5_tail, hit10_tail)


def save_model(model, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    torch.save({
        'model_state_dict': model.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def save_prediction(dataset, model, args, part):
    e2r_list = []
    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=dataset.collate_fn
    )
    batch_num = 0

    indexs = []
    topk = 5
    if args.dataset == 'WikiKG90Mv2':
        topk = 50
    for entity_pairs, labels, label_sets, tail_edges_lists, tail_masks, tail2relation in dataloader:
        if args.local_rank == -1:
            pos_score = model.test_step_score(
                get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda)
            )
        else:
            pos_score = model.module.test_step_score(
                get_feed_dict(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, args.cuda)
            )

        rows, cols, vals = [], [], []
        for idx in range(label_sets.shape[0]):
            sorted_indices = np.argsort(-pos_score[idx])
            if topk > 0:
                label_in_train_num = len(np.nonzero(label_sets[idx]))
                recall_num = 0
                for i in range(label_in_train_num + topk):
                    if pos_score[idx][i] == 0:
                        break
                recall_num = i
                cols.append(sorted_indices[:recall_num])
                rows.append([idx] * (recall_num))
                vals.append(pos_score[idx][cols[-1]])
            else:
                nonzeros = len(np.nonzero(pos_score[idx])[0])
                cols.append(sorted_indices[:nonzeros])
                rows.append([idx] * nonzeros)
                vals.append(pos_score[idx][cols[-1]])

        val = np.concatenate(vals)
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        e2r = coo_matrix((val, (row, col)), shape=pos_score.shape).tocsr()

        e2r_list.append(e2r)
        batch_num += 1
        if batch_num % 1000 == 0:
            print(batch_num)

    e2r = vstack(e2r_list)
    scipy.sparse.save_npz(os.path.join(args.save_path, ('e2r_scores_%d.npz' % part)), e2r)

    return


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.

    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.

    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function


def get_feed_dict_to_cuda(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, gpu_id):
    device = torch.device('cuda:' + str(gpu_id))
    feed_dict = {}
    feed_dict["labels"] = labels.to(device)
    feed_dict["label_sets"] = label_sets.to(device)
    feed_dict["tail_edges_lists"] = [t.to(device) for t in tail_edges_lists]
    feed_dict["tail2relation"] = [t.to(device) for t in tail2relation]
    if tail_masks is None:
        tail_masks = [torch.ones(t.size(), dtype=torch.bool) for t in tail2relation]
        feed_dict["tail_masks"] = [t.to(device) for t in tail_masks]
    else:
        feed_dict["tail_masks"] = [t.to(device) for t in tail_masks]

    return feed_dict


@thread_wrapped_func
def save_prediction_mp(dataloader, model, args, gpu_id, part):
    e2r_list = []
    batch_num = 0

    indexs = []
    topk = 5
    if args.dataset == 'WikiKG90Mv2':
        topk = 50
    # start1 = time.time()
    for entity_pairs, labels, label_sets, tail_edges_lists, tail_masks, tail2relation in dataloader:
        pos_score = model.test_step_score(
            get_feed_dict_to_cuda(labels, label_sets, tail_edges_lists, tail_masks, tail2relation, gpu_id)
        )
        # print("forward take {} seconds".format(time.time() - start1))

        rows, cols, vals = [], [], []
        # start1 = time.time()
        for idx in range(label_sets.shape[0]):
            sorted_indices = np.argsort(-pos_score[idx])
            if topk > 0:
                label_in_train_num = len(np.nonzero(label_sets[idx]))
                recall_num = 0
                for i in range(label_in_train_num + topk):
                    if pos_score[idx][i] == 0:
                        break
                recall_num = i
                cols.append(sorted_indices[:recall_num])
                rows.append([idx] * (recall_num))
                vals.append(pos_score[idx][cols[-1]])
            else:
                nonzeros = len(np.nonzero(pos_score[idx])[0])
                cols.append(sorted_indices[:nonzeros])
                rows.append([idx] * nonzeros)
                vals.append(pos_score[idx][cols[-1]])
        # print("process result {} seconds".format(time.time() - start1))
        #
        # start1 = time.time()
        val = np.concatenate(vals)
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        e2r = coo_matrix((val, (row, col)), shape=pos_score.shape).tocsr()

        e2r_list.append(e2r)
        # print("merge result {} seconds".format(time.time() - start1))
        batch_num += 1
        if batch_num % 1000 == 0:
            print(batch_num)

        # start1 = time.time()

    e2r = vstack(e2r_list)
    scipy.sparse.save_npz(os.path.join(args.save_path, ('e2r_scores_%d.npz' % part)), e2r)

    return

