from __future__ import print_function

import pickle
import argparse
import sys
import time
import os

import torch
import numpy as np

from utils import read_conll, \
                  to_input_tensor, \
                  data_iter, \
                  generate_seed, \
                  sents_to_vec


from markov_model import MarkovFlow


def init_config():

    parser = argparse.ArgumentParser(description='POS tagging')

    # train and test data
    parser.add_argument('--word_vec', type=str,
        help='the word vector file (cPickle saved file)')
    parser.add_argument('--train_file', type=str, help='train data')
    parser.add_argument('--test_file', default='', type=str, help='test data')

    # optimization parameters
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    # model config
    parser.add_argument('--model', choices=['gaussian', 'nice'], default='gaussian')
    parser.add_argument('--num_state', default=45, type=int,
        help='number of hidden states of z')
    parser.add_argument('--couple_layers', default=4, type=int,
        help='number of coupling layers in NICE')
    parser.add_argument('--cell_layers', default=1, type=int,
        help='number of cell layers of ReLU net in each coupling layer')
    parser.add_argument('--hidden_units', default=50, type=int, help='hidden units in ReLU Net')

    # pretrained model options
    parser.add_argument('--load_nice', default='', type=str,
        help='load pretrained projection model, ignored by default')
    parser.add_argument('--load_gaussian', default='', type=str,
        help='load pretrained Gaussian model, ignored by default')

    # log parameters
    parser.add_argument('--valid_nepoch', default=1, type=int, help='valid_nepoch')

    # Others
    parser.add_argument('--tag_from', default='', type=str,
        help='load pretrained model and perform tagging')
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--set_seed', action='store_true', default=False, help='if set seed')

    # these are for slurm purpose to save model
    # they can also be used to run multiple random restarts with various settings,
    # to save models that can be identified with ids
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "dump_models/markov"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "pos_%s_%dlayers_%d_%d" % (args.model, args.couple_layers, args.jobid, args.taskid)
    save_path = os.path.join(save_dir, id_ + '.pt')
    args.save_path = save_path
    print("model save path: ", save_path)

    if args.tag_from != '':
        if args.model == 'nice':
            args.load_nice = args.tag_from
        else:
            args.load_gaussian = args.tag_from
        args.tag_path = "pos_%s_%slayers_tagging%d_%d.txt" % \
        (args.model, args.couple_layers, args.jobid, args.taskid)

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

    train_text, null_index = read_conll(args.train_file)
    if args.test_file != '':
        test_text, null_index = read_conll(args.test_file)
    else:
        test_text = train_text

    train_data = sents_to_vec(word_vec, train_text)
    test_data = sents_to_vec(word_vec, test_text)

    test_tag = [sent["tag"] for sent in test_text]

    num_dims = len(train_data[0][0])
    print('complete reading data')

    print('#training sentences: %d' % len(train_data))
    print('#testing sentences: %d' % len(test_data))

    log_niter = (len(train_data)//args.batch_size)//10


    pad = np.zeros(num_dims)
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    init_seed = to_input_tensor(generate_seed(train_data, args.batch_size),
                                  pad, device=device)

    model = MarkovFlow(args, num_dims).to(device)

    model.init_params(init_seed)

    if args.tag_from != '':
        model.eval()
        with torch.no_grad():
            accuracy, vm = model.test(test_data, test_tags, sentences=test_text,
                tagging=True, path=args.tag_path, null_index=null_index)
        print('\n***** M1 %f, VM %f, max_var %.4f, min_var %.4f*****\n'
              % (accuracy, vm, model.var.data.max(), model.var.data.min()), file=sys.stderr)
        return


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    begin_time = time.time()
    print('begin training')

    train_iter = report_obj = report_jc = report_ll = report_num_words = 0

    # print the accuracy under init params
    model.eval()
    with torch.no_grad():
        accuracy, vm = model.test(test_data, test_tags)
    print('\n*****starting M1 %f, VM %f, max_var %.4f, min_var %.4f*****\n'
          % (accuracy, vm, model.var.data.max(), model.var.data.min()), file=sys.stderr)


    model.train()
    for epoch in range(args.epochs):
        # model.print_params()
        report_obj = report_jc = report_ll = report_num_words = 0
        for sents in data_iter(train_data, batch_size=args.batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(sents)
            num_words = sum(len(sent) for sent in sents)
            sents_var, masks = to_input_tensor(sents, pad, device=args.device)
            optimizer.zero_grad()
            likelihood, jacobian_loss = model(sents_var, masks)
            neg_likelihood_loss = -likelihood

            avg_ll_loss = (neg_likelihood_loss + jacobian_loss)/batch_size

            avg_ll_loss.backward()

            optimizer.step()

            log_likelihood_val = -neg_likelihood_loss.item()
            jacobian_val = -jacobian_loss.item()
            obj_val = log_likelihood_val + jacobian_val

            report_ll += log_likelihood_val
            report_jc += jacobian_val
            report_obj += obj_val
            report_num_words += num_words

            if train_iter % log_niter == 0:
                print('epoch %d, iter %d, log_likelihood %.2f, jacobian %.2f, obj %.2f, max_var %.4f ' \
                      'min_var %.4f time elapsed %.2f sec' % (epoch, train_iter, report_ll / report_num_words, \
                      report_jc / report_num_words, report_obj / report_num_words, model.var.max(), \
                      model.var.min(), time.time() - begin_time), file=sys.stderr)

        print('\nepoch %d, log_likelihood %.2f, jacobian %.2f, obj %.2f\n' % \
            (epoch, report_ll / report_num_words, report_jc / report_num_words,
             report_obj / report_num_words), file=sys.stderr)

        if epoch % args.valid_nepoch == 0:
            model.eval()
            with torch.no_grad():
                accuracy, vm = model.test(test_data, test_tags)
            print('\n*****epoch %d, iter %d, M1 %f, VM %f*****\n' %
                (epoch, train_iter, accuracy, vm), file=sys.stderr)
            model.train()

    model.eval()
    torch.save(model.state_dict(), args.save_path)
    with torch.no_grad():
        accuracy, vm = model.test(test_data, test_tags)
    print('\n complete training, accuracy %f, vm %f\n' % (accuracy, vm), file=sys.stderr)

if __name__ == '__main__':
    parse_args = init_config()
    main(parse_args)
