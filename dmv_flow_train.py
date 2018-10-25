from __future__ import print_function

import os
import argparse
import time
import sys
import pickle

import torch
import numpy as np

import dmv_flow_model as dmv
from utils import data_iter, \
                  read_conll, \
                  sents_to_vec, \
                  sents_to_tagid, \
                  to_input_tensor, \
                  generate_seed


def init_config():

    parser = argparse.ArgumentParser(description='dependency parsing')

    # train and test data
    parser.add_argument('--word_vec', type=str,
        help='the word vector file (cPickle saved file)')
    parser.add_argument('--train_file', type=str, help='train data')
    parser.add_argument('--test_file', default='', type=str, help='test data')
    parser.add_argument('--load_viterbi_dmv', type=str,
        help='load pretrained DMV')

    # optimization parameters
    parser.add_argument('--epochs', default=15, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')

    # model config
    parser.add_argument('--model', choices=['gaussian', 'nice'], default='gaussian')
    parser.add_argument('--couple_layers', default=4, type=int,
        help='number of coupling layers in NICE')
    parser.add_argument('--cell_layers', default=1, type=int,
        help='number of cell layers of ReLU net in each coupling layer')
    parser.add_argument('--hidden_units', default=50, type=int, help='hidden units in ReLU Net')    

    # others
    parser.add_argument('--train_from', type=str, default='',
        help='load a pre-trained checkpoint')
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--set_seed', action='store_true', default=False, 
        help='if set seed')
    parser.add_argument('--valid_nepoch', default=1, type=int, 
        help='valid every n epochs')   

    # these are for slurm purpose to save model
    # they can also be used to run multiple random restarts with various settings,
    # to save models that can be identified with ids
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id') 


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "dump_models/dmv"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = "parse_%s_%dlayers_%d_%d" % \
                (args.model, args.couple_layers, args.jobid, args.taskid)
    save_path = os.path.join(save_dir, save_path + '.pt')
    args.save_path = save_path

    if args.set_seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed * 13 / 7)

    print(args)

    return args


def main(args):

    word_vec = pickle.load(open(args.word_vec, 'rb'))
    print('complete loading word vectors')

    train_sents, _ = read_conll(args.train_file)
    test_sents, _ = read_conll(args.test_file, max_len=10)
    test_deps = [sent["head"] for sent in test_sents]

    train_emb = sents_to_vec(word_vec, train_sents)
    test_emb = sents_to_vec(word_vec, test_sents)

    num_dims = len(train_emb[0][0])

    train_tagid, tag2id = sents_to_tagid(train_sents)
    print('%d types of tags' % len(tag2id))
    id2tag = {v: k for k, v in tag2id.items()}

    pad = np.zeros(num_dims)
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    model = dmv.DMVFlow(args, id2tag, num_dims).to(device)

    init_seed = to_input_tensor(generate_seed(train_emb, args.batch_size),
                                pad, device=device)

    with torch.no_grad():
        model.init_params(init_seed, train_tagid, train_emb)
    print('complete init')

    if args.train_from != '':
        model.load_state_dict(torch.load(args.train_from))
        with torch.no_grad():
            directed, undirected = model.test(test_deps, test_emb, verbose=False)
        print('acc on length <= 10: #trees %d, undir %2.1f, dir %2.1f' \
              % (len(test_gold), 100 * undirected, 100 * directed))

        # directed, undirected = model.eval(test_gold_full, test_emb_full, verbose=True)
        # print('accuracy on all lengths: number of trees:%d, undir: %2.1f, dir: %2.1f' \
        #       % (len(test_gold), 100 * undirected, 100 * directed))
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_niter = (len(train_emb)//args.batch_size)//5
    report_ll = report_num_words = report_num_sents = epoch = train_iter = 0
    stop_avg_ll = stop_num_words = 0
    stop_avg_ll_last = 1
    dir_last = 0
    begin_time = time.time()

    print('begin training')

    with torch.no_grad():
        directed, undirected = model.test(test_deps, test_emb)
    print('starting acc on length <= 10: #trees %d, undir %2.1f, dir %2.1f' \
          % (len(test_deps), 100 * undirected, 100 * directed))

    for epoch in range(args.epochs):
        report_ll = report_num_sents = report_num_words = 0
        for sents in data_iter(train_emb, batch_size=args.batch_size):
            batch_size = len(sents)
            num_words = sum(len(sent) for sent in sents)
            stop_num_words += num_words
            optimizer.zero_grad()

            sents_var, masks = to_input_tensor(sents, pad, device)
            sents_var, _ = model.transform(sents_var)
            sents_var = sents_var.transpose(0, 1)
            # log_likelihood = sum([model.p_inside(sent) for sent in sents])
            log_likelihood = model.p_inside(sents_var, masks)

            avg_ll_loss = -log_likelihood / batch_size

            avg_ll_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            report_ll += log_likelihood.item()
            report_num_words += num_words
            report_num_sents += batch_size

            stop_avg_ll += log_likelihood.item()

            if train_iter % log_niter == 0:
                print('epoch %d, iter %d, ll_per_sent %.4f, ll_per_word %.4f, ' \
                      'max_var %.4f, min_var %.4f time elapsed %.2f sec' % \
                      (epoch, train_iter, report_ll / report_num_sents, \
                      report_ll / report_num_words, model.var.data.max(), \
                      model.var.data.min(), time.time() - begin_time), file=sys.stderr)

            train_iter += 1
        # model.print_params()
        if epoch % args.valid_nepoch == 0:
            with torch.no_grad():
                directed, undirected = model.test(test_deps, test_emb)
            print('\n\nacc on length <= 10: #trees %d, undir %2.1f, dir %2.1f, \n\n' \
                  % (len(test_deps), 100 * undirected, 100 * directed))

        stop_avg_ll = stop_avg_ll / stop_num_words
        rate = (stop_avg_ll - stop_avg_ll_last) / abs(stop_avg_ll_last)

        print('\n\nlikelihood: %.4f, likelihood last: %.4f, rate: %f\n' % \
                (stop_avg_ll, stop_avg_ll_last, rate))

        if rate < 0.001 and epoch >= 5:
            break

        stop_avg_ll_last = stop_avg_ll
        stop_avg_ll = stop_num_words = 0

    # model.load_state_dict(torch.load(args.save_path))
    # eval on all lengths
    # directed, undirected = model.eval(test_gold_full, test_emb_full, verbose=True)
    # print('accuracy on all lengths: number of trees:%d, undir: %2.1f, dir: %2.1f' \
    #       % (len(test_gold), 100 * undirected, 100 * directed))

    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parse_args = init_config()
    main(parse_args)
