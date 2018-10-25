from __future__ import print_function
import math
import pickle
import torch
import torch.nn as nn

import numpy as np

from collections import Counter
from projection import NICETrans
from dmv_viterbi_model import DMVDict

from torch.nn import Parameter

from utils import log_sum_exp, \
                  unravel_index, \
                  data_iter, \
                  to_input_tensor, \
                  stable_math_log

harmonic_constant = 2.0
NEG_INFINITY = -1e20

def test_piodict(piodict):
    """
    test PIOdict 0 value

    """
    for key, value in piodict.dict.iteritems():
        if value <= 0:
            print(key, value)
            return False
    return True

def log_softmax(input, dim):
    return (input - \
            log_sum_exp(input, dim=dim, keepdim=True) \
            .expand_as(input))


class DMVFlow(nn.Module):
    def __init__(self, args, ids, num_dims):
        super(DMVFlow, self).__init__()

        self.ids = ids
        self.num_state = len(ids)
        self.num_dims = num_dims
        self.args = args
        self.device = args.device

        self.harmonic = False

        self.means = Parameter(torch.Tensor(self.num_state, self.num_dims))

        if args.model == 'nice':
            self.nice_layer = NICETrans(self.args.couple_layers,
                                        self.args.cell_layers,
                                        self.args.hidden_units,
                                        self.num_dims,
                                        self.device)


        # Gaussian Variance
        self.var = torch.zeros(num_dims, dtype=torch.float32,
            device=self.device, requires_grad=False)

        # dim0 is head and dim1 is argument
        self.attach_left = Parameter(torch.Tensor(self.num_state, self.num_state))
        self.attach_right = Parameter(torch.Tensor(self.num_state, self.num_state))

        # (stop, adj, h)
        # dim0: 0 is nonstop, 1 is stop
        # dim2: 0 is nonadjacent, 1 is adjacent
        self.stop_right = Parameter(torch.Tensor(2, self.num_state, 2))
        self.stop_left = Parameter(torch.Tensor(2, self.num_state, 2))

        self.root_attach_left = Parameter(torch.Tensor(self.num_state))

    def init_params(self, init_seed, train_tagid, train_emb):
        """
        init_seed:(sents, masks)
        sents: (seq_length, batch_size, features)
        masks: (seq_length, batch_size)

        """

        # init transition params
        self.attach_left.uniform_().add_(0.01)
        self.attach_right.uniform_().add_(0.01)
        self.root_attach_left.uniform_().add_(1)

        self.stop_right[0, :, 0].uniform_().add_(1)
        self.stop_right[1, :, 0].uniform_().add_(2)
        self.stop_left[0, :, 0].uniform_().add_(1)
        self.stop_left[1, :, 0].uniform_().add_(2)

        self.stop_right[0, :, 1].uniform_().add_(2)
        self.stop_right[1, :, 1].uniform_().add_(1)
        self.stop_left[0, :, 1].uniform_().add_(2)
        self.stop_left[1, :, 1].uniform_().add_(1)

        # initialize mean and variance with empirical values
        sents, masks = init_seed
        sents, _ = self.transform(sents)
        features = sents.size(-1)
        flat_sents = sents.view(-1, features)
        seed_mean = torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                              flat_sents, dim=0) / masks.sum()
        seed_var = torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                             ((flat_sents - seed_mean.expand_as(flat_sents)) ** 2),
                             dim=0) / masks.sum()

        self.var.copy_ = seed_var
        self.init_mean(train_tagid, train_emb)

        load_model = pickle.load(open(self.args.load_viterbi_dmv, 'rb'))
        for i in range(self.num_state):
            for j in range(self.num_state):
                self.attach_left[i, j] = load_model.tita.val(
                                              ('attach_left', self.ids[j], self.ids[i]))

                self.attach_right[i, j] = load_model.tita.val(
                                               ('attach_right', self.ids[j], self.ids[i]))

            self.stop_left[1, i, 0] = load_model.tita \
                                           .val(('stop_left', self.ids[i], 0))
            self.stop_left[0, i, 0] = stable_math_log(1.0 - math.exp(load_model.tita \
                                           .val(('stop_left', self.ids[i], 0))))
            self.stop_left[1, i, 1] = load_model.tita \
                                           .val(('stop_left', self.ids[i], 1))
            self.stop_left[0, i, 1] = stable_math_log(1.0 - math.exp(load_model.tita \
                                           .val(('stop_left', self.ids[i], 1))))
            self.stop_right[1, i, 0] = load_model.tita \
                                           .val(('stop_right', self.ids[i], 0))
            self.stop_right[0, i, 0] = stable_math_log(1.0 - math.exp(load_model.tita \
                                           .val(('stop_right', self.ids[i], 0))))
            self.stop_right[1, i, 1] = load_model.tita \
                                           .val(('stop_right', self.ids[i], 1))
            self.stop_right[0, i, 1] = stable_math_log(1 - math.exp(load_model.tita \
                                           .val(('stop_right', self.ids[i], 1))))
            self.root_attach_left[i] = load_model.tita \
                                            .val(('attach_left', self.ids[i], 'END'))

    def init_mean(self, train_tagid, train_emb):
        pad = np.zeros(self.num_dims)
        emb_dict = {}
        cnt_dict = Counter()
        for sents, tagid_sents in data_iter(list(zip(train_emb, train_tagid)), \
                                           batch_size=self.args.batch_size, \
                                           is_test=True, \
                                           shuffle=False):
            sents_var, masks = to_input_tensor(sents, pad, self.device)
            sents_var, _ = self.transform(sents_var)
            sents_var = sents_var.transpose(0, 1)
            for tagid_sent, emb_sent in zip(tagid_sents, sents_var):
                for tagid, emb in zip(tagid_sent, emb_sent):
                    if tagid in emb_dict:
                        emb_dict[tagid] = emb_dict[tagid] + emb
                    else:
                        emb_dict[tagid] = emb

                    cnt_dict[tagid] += 1

        for i in range(self.num_state):
            self.means[i] = emb_dict[i] / cnt_dict[i]


    def transform(self, x):
        """
        Args:
            x: (sent_length, batch_size, num_dims)
        """
        jacobian_loss = torch.zeros(1, device=self.device, requires_grad=False)

        if self.args.model == 'nice':
            x, jacobian_loss_new = self.nice_layer(x)
            jacobian_loss = jacobian_loss + jacobian_loss_new


        return x, jacobian_loss

    def tree_to_depset(self, root_max_index, sent_len):
        """
        Args:
            root_max_index: (batch_size, 2)
        """
        # add the root symbol (-1)
        batch_size = root_max_index.size(0)
        dep_list = []
        for batch in range(batch_size):
            res = set([(root_max_index[batch, 1], -1)])
            start = 0
            end = sent_len[batch]
            res.update(self._tree_to_depset(start, end, 2, batch, root_max_index[batch, 0],
                                            root_max_index[batch, 1]))
            assert len(res) == sent_len[batch]
            dep_list += [depset.DepSet(sent_len[batch], sorted(res))]

        return dep_list

    def _tree_to_depset(self, start, end, mark, batch, symbol, index):
        left_child = self.left_child[start, end, mark][batch, symbol, index]
        right_child = self.right_child[start, end, mark][batch, symbol, index]

        if left_child[0] == 1 and right_child[0] == 1:
            if mark == 0:
                assert left_child[3] == 0
                assert right_child[3] == 2
                arg = right_child[-1]
            elif mark == 1:
                assert left_child[3] == 2
                assert right_child[3] == 1
                arg = left_child[-1]
            res = set([(arg, index)])
            res.update(self._tree_to_depset(left_child[1], left_child[2], left_child[3], \
                                           batch, left_child[4], left_child[5]), \
                       self._tree_to_depset(right_child[1], right_child[2], right_child[3], \
                                           batch, right_child[4], right_child[5]))

        elif left_child[0] == 1 and right_child[0] == 0:
            res = self._tree_to_depset(left_child[1], left_child[2], left_child[3], \
                                      batch, left_child[4], left_child[5])
        elif left_child[0] == -1 and right_child[0] == -1:
            res = set()

        else:
            raise ValueError

        return res

    def test(self, gold, test_emb, all_len=False):
        """
        Args:
            gold: A nested list of heads
            all_len: True if evaluate on all lengths
        """
        pad = np.zeros(self.num_dims)
        cnt = 0
        dir_cnt = 0.0
        undir_cnt = 0.0
        memory_sent_cnt = 0

        batch_id_ = 0

        if all_len:
            batch_size = 2
        else:
            batch_size = self.args.batch_size

        for sents, gold_batch in data_iter(list(zip(test_emb, gold)), 
                                           batch_size=batch_size,
                                           is_test=True, 
                                           shuffle=False):

            if all_len:
                print('batch %d' % batch_id_)
            batch_id_ += 1
            try:
                sents_var, masks = to_input_tensor(sents, pad, self.device)
                sents_var, _ = self.transform(sents_var)
                sents_var = sents_var.transpose(0, 1)
                # root_max_index: (batch_size, num_state, seq_length)
                batch_size, seq_length, _ = sents_var.size()
                symbol_index_t = self.attach_left.new_tensor([[[p, q] for q in range(seq_length)] \
                                                      for p in range(self.num_state)]) \
                                                      .expand(batch_size, self.num_state, seq_length, 2)
                root_max_index = self.dep_parse(sents_var, masks, symbol_index_t)
                batch_size = masks.size(1)
                sent_len = [torch.sum(masks[:, i]) for i in range(batch_size)]
                parse = self.tree_to_depset(root_max_index, sent_len)
            except RuntimeError:
                memory_sent_cnt += 1
                print('batch %d out of memory' % batch_id_)
                continue

            for gold_s, parse_s in zip(gold_batch, parse):
                assert len(gold_s) == len(parse_s)
                # assert gold_s.length > 1
                if len(gold_s) > 1:
                    (directed, undirected) = self.measures(gold_s, parse_s)
                    cnt += length
                    dir_cnt += directed
                    undir_cnt += undirected

            if all_len:
                print('total length: %d' % cnt)
                print('correct directed: %d' % dir_cnt)

        dir_acu = dir_cnt / cnt
        undir_acu = undir_cnt / cnt

        self.log_p_parse = {}
        self.left_child = {}
        self.right_child = {}

        if all_len:
            print('%d batches out of memory' % memory_sent_cnt)

        return (dir_acu, undir_acu)

    @staticmethod
    def measures(gold_s, parse_s):
        # Helper for eval().
        (d, u) = (0, 0)
        for (a, b) in gold_s:
            (a, b) = (a-1, b-1)
            b1 = (a, b) in parse_s
            b2 = (b, a) in parse_s
            if b1:
                d += 1.0
                u += 1.0
            if b2:
                u += 1.0

        return (d, u)


    def _eval_log_density(self, s):
        """
        Args:
            s: A tensor with size (batch_size, seq_length, features)

        Returns:
            density: (batch_size, seq_length, num_state)

        """
        batch_size, seq_length, features = s.size()
        ep_size = torch.Size([batch_size, seq_length, self.num_state, features])
        means = self.means.view(1, 1, self.num_state, features).expand(ep_size)
        words = s.unsqueeze(dim=2).expand(ep_size)
        var = self.var.expand(ep_size)
        return self.log_density_c - \
               0.5 * torch.sum((means - words) ** 2 / var, dim=3)

    def _calc_log_density_c(self):

        return -self.num_dims/2.0 * (math.log(2 * math.pi)) - \
                0.5 * torch.sum(torch.log(self.var))

    def p_inside(self, sents, masks):
        """
        Args:
            sents: A tensor with size (batch_size, seq_length, features)
            masks: (seq_length, batch_size)

        Variable clarification:
            p_inside[i, j] is the prob of w_i, w_i+1, ..., w_j-1
            rooted at any possible nonterminals

        node marks clarification:
            '>': no marks (right first)
            '<>': right stop mark
            '|': both left and right stop marks

        """

        self.log_density_c = self._calc_log_density_c()

        # normalizing parameters
        self.log_attach_left = log_softmax(self.attach_left, dim=1)
        self.log_attach_right = log_softmax(self.attach_right, dim=1)
        self.log_stop_right = log_softmax(self.stop_right, dim=0)
        self.log_stop_left = log_softmax(self.stop_left, dim=0)
        self.log_root_attach_left = log_softmax(self.root_attach_left, dim=0)

        # (batch_size, seq_length, num_state)
        density = self._eval_log_density(sents)

        # indexed by (start, end, mark)
        # each element is a tensor with size (batch_size, num_state, seq_length)
        self.log_p_inside = {}
        # n = len(s)

        #TODO(junxian): batch sentences
        batch_size, seq_length, _ = sents.size()

        # if self.args.cuda:
        #     unary_index = torch.cuda.LongTensor(self.num_state, seq_length).zero_()
        # else:
        #     unary_index = torch.LongTensor(self.num_state, seq_length).zero_()

        for i in range(seq_length):
            j = i + 1
            if self.args.cuda:
                cat_var = [Variable(torch.cuda.FloatTensor(batch_size, self.num_state, 1) \
                           .fill_(NEG_INFINITY)) for _ in range(seq_length)]
            else:
                cat_var = [Variable(torch.FloatTensor(batch_size, self.num_state, 1) \
                           .fill_(NEG_INFINITY)) for _ in range(seq_length)]

            cat_var[i] = density[:, i, :].unsqueeze(dim=2)
            self.log_p_inside[i, j, 0] = torch.cat(cat_var, dim=2)
            self.unary_p_inside(i, j, batch_size, seq_length)

        ep_size = torch.Size([batch_size, self.num_state, seq_length, self.num_state, seq_length])
        log_attach_right = self.log_attach_right.view(1, self.num_state, 1, self.num_state, 1) \
                       .expand(ep_size)
        log_stop_right = self.log_stop_right[0].view(1, self.num_state, 2, 1, 1) \
                     .expand(batch_size, self.num_state, 2, self.num_state, seq_length)
        log_attach_left = self.log_attach_left.view(1, self.num_state, 1, self.num_state, 1) \
                      .transpose(1, 3).expand(ep_size)
        log_stop_left = self.log_stop_left[0].view(1, 1, 1, self.num_state, 2) \
                     .expand(batch_size, self.num_state, seq_length, self.num_state, 2)

        if self.args.cuda:
            index = torch.cuda.LongTensor(ep_size).zero_()
        else:
            index = torch.LongTensor(ep_size).zero_()
        #TODO(junxian): ideally, only the l loop is needed
        # but eliminate the rest loops would be a bit hard
        for l in range(2, seq_length+1):
            for i in range(seq_length-l+1):
                j = i + l
                p_inside_right = []
                p_inside_left = []
                for k in range(i+1, j):

                    # right attachment
                    log_p1 = self.log_p_inside[i, k, 0]
                    log_p2 = self.log_p_inside[k, j, 2]
                    log_p1_ep = log_p1.view(batch_size, self.num_state, seq_length, 1, 1) \
                                .expand(ep_size)
                    log_p2_ep = log_p2.view(batch_size, 1, 1, self.num_state, seq_length) \
                                .expand(ep_size)


                    index.zero_()
                    index[:, :, k-1, :, :].fill_(1)
                    index_var = Variable(index)
                    log_stop_right_gather = torch.gather(log_stop_right, 2, index_var)

                    # log_p_tmp[b, i, m, j, n] = log_p1[b, i, m] + log_p2[b, j, n] + stop_right[0, m==k-1, i]
                    # + attach_right[i, j]
                    log_p_tmp = log_p1_ep + log_p2_ep + log_attach_right + log_stop_right_gather

                    # log_p_tmp[i, m, j, n]
                    # log_p_tmp = log_p_tmp.transpose(1, 2)
                    p_inside_right.append(log_p_tmp)


                    # left attachment
                    log_p1 = self.log_p_inside[i, k, 2]
                    log_p2 = self.log_p_inside[k, j, 1]
                    log_p1_ep = log_p1.view(batch_size, self.num_state, seq_length, 1, 1) \
                                .expand(ep_size)
                    log_p2_ep = log_p2.view(batch_size, 1, 1, self.num_state, seq_length) \
                                .expand(ep_size)

                    index.zero_()

                    index[:, :, :, :, k].fill_(1)
                    index_var = Variable(index)
                    log_stop_left_gather = torch.gather(log_stop_left, 4, index_var)

                    # log_p_tmp[b, i, m, j, n] = log_p1[b, i, m] + log_p2[b, j, n] + stop_left[0, n==k, j]
                    # + self.attach_left[j, i]
                    log_p_tmp = log_p1_ep + log_p2_ep + log_attach_left + log_stop_left_gather

                    # log_p_tmp[b, i, m, j, n]
                    # log_p_tmp = log_p_tmp.permute(1, 3, 0, 2)
                    p_inside_left.append(log_p_tmp)

                # TODO(junxian): ensure the invalid index has -inf value
                # TODO(junxian): in_place
                self.log_p_inside[i, j, 0] = log_sum_exp(torch.cat(p_inside_right, dim=-1) \
                                                   .view(batch_size, self.num_state, seq_length, -1), dim=3)

                self.log_p_inside[i, j, 1] = log_sum_exp(torch.cat(p_inside_left, dim=1) \
                                                   .view(batch_size, -1, self.num_state, seq_length), dim=1)

                self.unary_p_inside(i, j, batch_size, seq_length)


        # calculate log likelihood
        sent_len_t = masks.sum(dim=0).detach()
        log_p_sum = []
        for i in range(batch_size):
            sent_len = sent_len_t[i]
            log_p_sum += [self.log_p_inside[0, sent_len, 2][i].unsqueeze(dim=0)]
        log_p_sum_cat = torch.cat(log_p_sum, dim=0)

        log_root = log_p_sum_cat + self.log_root_attach_left.view(1, self.num_state, 1) \
                   .expand_as(log_p_sum_cat)

        return torch.sum(log_sum_exp(log_root.view(batch_size, -1), dim=1))


    def dep_parse(self, sents, masks, symbol_index_t):
        """
        Args:
            sents: tensor with size (batch_size, seq_length, features)
        Returns:
            returned t is a nltk.tree.Tree without root node
        """

        self.log_density_c = self._calc_log_density_c()

        # normalizing parameters
        self.log_attach_left = log_softmax(self.attach_left, dim=1)
        self.log_attach_right = log_softmax(self.attach_right, dim=1)
        self.log_stop_right = log_softmax(self.stop_right, dim=0)
        self.log_stop_left = log_softmax(self.stop_left, dim=0)
        self.log_root_attach_left = log_softmax(self.root_attach_left, dim=0)


        # (batch_size, seq_length, num_state)
        density = self._eval_log_density(sents)


        # in the parse case, log_p_parse[i, j, mark] is not the log prob
        # of some symbol as head, instead it is the prob of the most likely
        # subtree with some symbol as head
        self.log_p_parse = {}

        # child is indexed by (i, j, mark), and each element is a
        # LongTensor with size (batch_size, symbol, seq_length, 6)
        # the last dimension represents the child's
        # (indicator, i, j, mark, symbol, index), used to index the child,
        # indicator is 1 represents childs exist, 0 not exist, -1 means
        # reaching terminal symbols. For unary connection, left child indicator
        # is 1 and right child indicator is 0 (for non-terminal symbols)
        self.left_child = {}
        self.right_child = {}

        batch_size, seq_length, _ = sents.size()

        ep_size = torch.Size([batch_size, self.num_state, seq_length, \
                              self.num_state, seq_length])

        if self.args.cuda:
            index = torch.cuda.LongTensor(ep_size).zero_()
        else:
            index = torch.LongTensor(ep_size).zero_()

        for i in range(seq_length):
            j = i + 1
            if self.args.cuda:
                cat_var = [torch.cuda.FloatTensor(batch_size, self.num_state, 1).fill_(NEG_INFINITY) \
                           for _ in range(seq_length)]
            else:
                cat_var = [torch.FloatTensor(batch_size, self.num_state, 1).fill_(NEG_INFINITY) \
                           for _ in range(seq_length)]
            cat_var[i] = density[:, i, :].unsqueeze(dim=2)
            self.log_p_parse[i, j, 0] = torch.cat(cat_var, dim=2)
            self.left_child[i, j, 0] = index.new(batch_size, self.num_state, seq_length, 6) \
                                            .fill_(-1)
            self.right_child[i, j, 0] = index.new(batch_size, self.num_state, seq_length, 6) \
                                             .fill_(-1)
            self.unary_parses(i, j, batch_size, seq_length, symbol_index_t)

        log_attach_right = self.log_attach_right.view(1, self.num_state, 1, self.num_state, 1) \
                               .expand(ep_size)
        log_stop_right = self.log_stop_right[0].view(1, self.num_state, 2, 1, 1) \
                             .expand(batch_size, self.num_state, 2, self.num_state, seq_length)
        log_attach_left = self.log_attach_left.view(1, self.num_state, 1, self.num_state, 1) \
                              .transpose(1, 3).expand(ep_size)
        log_stop_left = self.log_stop_left[0].view(1, 1, 1, self.num_state, 2) \
                            .expand(batch_size, self.num_state, seq_length, self.num_state, 2)

        # ideally, only the l loop is needed
        # but eliminate the rest loops would be a bit hard
        for l in range(2, seq_length+1):
            for i in range(seq_length-l+1):
                j = i + l
                p_inside_right = []
                p_inside_left = []
                for k in range(i+1, j):

                    # right attachment
                    log_p1 = self.log_p_parse[i, k, 0]
                    log_p2 = self.log_p_parse[k, j, 2]
                    log_p1_ep = log_p1.view(batch_size, self.num_state, seq_length, 1, 1) \
                                .expand(ep_size)
                    log_p2_ep = log_p2.view(batch_size, 1, 1, self.num_state, seq_length) \
                                .expand(ep_size)


                    index.zero_()
                    index[:, :, k-1, :, :].fill_(1)
                    log_stop_right_gather = torch.gather(log_stop_right, 2, index)

                    # log_p_tmp[b, i, m, j, n] = log_p1[b, i, m] + log_p2[b, j, n] + stop_right[0, m==k-1, i]
                    # + attach_right[i, j]
                    log_p_tmp = log_p1_ep + log_p2_ep + log_attach_right + log_stop_right_gather

                    # log_p_tmp[b, i, m, j, n]
                    # log_p_tmp = log_p_tmp.transpose(1, 2)
                    p_inside_right.append(log_p_tmp.unsqueeze(dim=3))


                    # left attachment
                    log_p1 = self.log_p_parse[i, k, 2]
                    log_p2 = self.log_p_parse[k, j, 1]
                    log_p1_ep = log_p1.view(batch_size, self.num_state, seq_length, 1, 1) \
                                .expand(ep_size)
                    log_p2_ep = log_p2.view(batch_size, 1, 1, self.num_state, seq_length) \
                                .expand(ep_size)

                    index.zero_()

                    index[:, :, :, :, k].fill_(1)
                    log_stop_left_gather = torch.gather(log_stop_left, 4, index)

                    # log_p_tmp[b, i, m, j, n] = log_p1[b, i, m] + log_p2[b, j, n] + stop_left[0, n==k, j]
                    # + self.attach_left[j, i]
                    log_p_tmp = log_p1_ep + log_p2_ep + log_attach_left + log_stop_left_gather

                    # log_p_tmp[b, i, m, j, n]
                    p_inside_left.append(log_p_tmp.unsqueeze(dim=1))


                # log_p_max: (batch_size, num_state, seq_length)
                # max_index: (batch_size, num_state, seq_length)
                log_p_max_r, max_index_r = torch.max(torch.cat(p_inside_right, dim=3) \
                                           .view(batch_size, self.num_state, seq_length, -1), dim=-1)
                log_p_max_l, max_index_l = torch.max(torch.cat(p_inside_left, dim=1) \
                                           .view(batch_size, -1, self.num_state, seq_length), dim=1)

                # max_index: (batch_size, num_state, seq_length, 3)
                max_index_r = unravel_index(max_index_r, (j-i-1, self.num_state, seq_length))
                max_index_l = unravel_index(max_index_l, (j-i-1, self.num_state, seq_length))

                self.log_p_parse[i, j, 0] = log_p_max_r
                self.log_p_parse[i, j, 1] = log_p_max_l

                right_child_index_r = index.new(batch_size, self.num_state, seq_length, 6)
                left_child_index_r = index.new(batch_size, self.num_state, seq_length, 6)
                right_child_index_l = index.new(batch_size, self.num_state, seq_length, 6)
                left_child_index_l = index.new(batch_size, self.num_state, seq_length, 6)
                # assign symbol and index
                right_child_index_r[:, :, :, 4:] = max_index_r[:, :, :, 1:]

                # left_child_symbol_index: (num_state, seq_length, 2)
                left_child_symbol_index_r = symbol_index_t

                left_child_index_r[:, :, :, 4:] = left_child_symbol_index_r

                right_child_symbol_index_l = symbol_index_t

                right_child_index_l[:, :, :, 4:] = right_child_symbol_index_l
                left_child_index_l[:, :, :, 4:] = max_index_l[:, :, :, 1:]

                # assign indicator
                right_child_index_r[:, :, :, 0] = 1
                left_child_index_r[:, :, :, 0] = 1

                right_child_index_l[:, :, :, 0] = 1
                left_child_index_l[:, :, :, 0] = 1

                # assign starting point
                right_child_index_r[:, :, :, 1] = max_index_r[:, :, :, 0] + i + 1
                left_child_index_r[:, :, :, 1] = i

                right_child_index_l[:, :, :, 1] = max_index_l[:, :, :, 0] + i + 1
                left_child_index_l[:, :, :, 1] = i

                # assign end point
                right_child_index_r[:, :, :, 2] = j
                left_child_index_r[:, :, :, 2] = max_index_r[:, :, :, 0] + i + 1

                right_child_index_l[:, :, :, 2] = j
                left_child_index_l[:, :, :, 2] = max_index_l[:, :, :, 0] + i + 1

                right_child_index_r[:, :, :, 3] = 2
                left_child_index_r[:, :, :, 3] = 0

                right_child_index_l[:, :, :, 3] = 1
                left_child_index_l[:, :, :, 3] = 2

                assert (i, j, 0) not in self.left_child
                self.left_child[i, j, 0] = left_child_index_r
                self.right_child[i, j, 0] = right_child_index_r

                self.left_child[i, j, 1] = left_child_index_l
                self.right_child[i, j, 1] = right_child_index_l

                self.unary_parses(i, j, batch_size, seq_length, symbol_index_t)

        log_p_sum = []
        sent_len_t = masks.sum(dim=0)
        for i in range(batch_size):
            sent_len = sent_len_t[i]
            log_p_sum += [self.log_p_parse[0, sent_len, 2][i].unsqueeze(dim=0)]
        log_p_sum_cat = torch.cat(log_p_sum, dim=0)
        log_root = log_p_sum_cat + self.log_root_attach_left.view(1, self.num_state, 1) \
                   .expand_as(log_p_sum_cat)
        log_root_max, root_max_index = torch.max(log_root.view(batch_size, -1), dim=1)

        # (batch_size, 2)
        root_max_index = unravel_index(root_max_index, (self.num_state, seq_length))

        return root_max_index

    def unary_p_inside(self, i, j, batch_size, seq_length):

        non_stop_mark = self.log_p_inside[i, j, 0]
        log_stop_left = self.log_stop_left[1].expand(batch_size, self.num_state, 2)
        log_stop_right = self.log_stop_right[1].expand(batch_size, self.num_state, 2)
        if self.args.cuda:
            index_ladj = torch.cuda.LongTensor(batch_size, self.num_state, seq_length).zero_()
            index_radj = torch.cuda.LongTensor(batch_size, self.num_state, seq_length).zero_()
        else:
            index_ladj = torch.LongTensor(batch_size, self.num_state, seq_length).zero_()
            index_radj = torch.LongTensor(batch_size, self.num_state, seq_length).zero_()

        index_ladj[:, :, i].fill_(1)
        index_radj[:, :, j-1].fill_(1)

        index_ladj = Variable(index_ladj)
        index_radj = Variable(index_radj)

        log_stop_right = torch.gather(log_stop_right, 2, index_radj)
        inter_right_stop_mark = non_stop_mark + log_stop_right

        if (i, j, 1) in self.log_p_inside:
            right_stop_mark = self.log_p_inside[i, j, 1]
            right_stop_mark = torch.cat((right_stop_mark.unsqueeze(dim=3), \
                                    inter_right_stop_mark.unsqueeze(dim=3)), \
                                    dim=3)
            right_stop_mark = log_sum_exp(right_stop_mark, dim=3)

        else:
            right_stop_mark = inter_right_stop_mark



        log_stop_left = torch.gather(log_stop_left, 2, index_ladj)
        self.log_p_inside[i, j, 2] = right_stop_mark + log_stop_left
        self.log_p_inside[i, j, 1] = right_stop_mark

    def unary_parses(self, i, j, batch_size, seq_length, symbol_index_t):
        non_stop_mark = self.log_p_parse[i, j, 0]
        log_stop_left = self.log_stop_left[1].expand(batch_size, self.num_state, 2)
        log_stop_right = self.log_stop_right[1].expand(batch_size, self.num_state, 2)

        if self.args.cuda:
            index_ladj = torch.cuda.LongTensor(batch_size, self.num_state, seq_length).zero_()
            index_radj = torch.cuda.LongTensor(batch_size, self.num_state, seq_length).zero_()
        else:
            index_ladj = torch.LongTensor(batch_size, self.num_state, seq_length).zero_()
            index_radj = torch.LongTensor(batch_size, self.num_state, seq_length).zero_()

        left_child_index_mark2 = index_ladj.new(batch_size, self.num_state, seq_length, 6)
        right_child_index_mark2 = index_ladj.new(batch_size, self.num_state, seq_length, 6)
        left_child_index_mark1 = index_ladj.new(batch_size, self.num_state, seq_length, 6)
        right_child_index_mark1 = index_ladj.new(batch_size, self.num_state, seq_length, 6)

        index_ladj[:, :, i].fill_(1)
        index_radj[:, :, j-1].fill_(1)

        log_stop_right = torch.gather(log_stop_right, 2, index_radj)
        inter_right_stop_mark = non_stop_mark + log_stop_right

        # assign indicator
        left_child_index_mark1[:, :, :, 0] = 1
        right_child_index_mark1[:, :, :, 0] = 0

        # assign mark
        left_child_index_mark1[:, :, :, 3] = 0
        right_child_index_mark1[:, :, :, 3] = 0

        # start point
        left_child_index_mark1[:, :, :, 1] = i
        right_child_index_mark1[:, :, :, 1] = i

        # end point
        left_child_index_mark1[:, :, :, 2] = j
        right_child_index_mark1[:, :, :, 2] = j

        # assign symbol and index
        left_child_symbol_index_mark1 = symbol_index_t
        left_child_index_mark1[:, :, :, 4:] = left_child_symbol_index_mark1
        right_child_index_mark1[:, :, :, 4:] = left_child_symbol_index_mark1

        if (i, j, 1) in self.log_p_parse:
            right_stop_mark = self.log_p_parse[i, j, 1]

            # max_index (batch_size, num_state, index) (value is 0 or 1)
            right_stop_mark, max_index = torch.max(torch.cat((right_stop_mark.unsqueeze(dim=3), \
                                    inter_right_stop_mark.unsqueeze(dim=3)), \
                                    dim=3), dim=3)

            # mask: (batch_size, num_state, index)
            mask = (max_index == 1)
            mask_ep = mask.unsqueeze(dim=-1).expand(batch_size, self.num_state, seq_length, 6)
            left_child_index_mark1 = self.left_child[i, j, 1].masked_fill_(mask_ep, 0) + \
                                     left_child_index_mark1.masked_fill_(1 - mask_ep, 0)
            right_child_index_mark1 = self.right_child[i, j, 1].masked_fill_(mask_ep, 0) + \
                                     right_child_index_mark1.masked_fill_(1 - mask_ep, 0)


        else:
            right_stop_mark = inter_right_stop_mark

        log_stop_left = torch.gather(log_stop_left, 2, index_ladj)
        self.log_p_parse[i, j, 2] = right_stop_mark + log_stop_left
        self.log_p_parse[i, j, 1] = right_stop_mark


        # assign indicator
        left_child_index_mark2[:, :, :, 0] = 1
        right_child_index_mark2[:, :, :, 0] = 0

        # assign starting point
        left_child_index_mark2[:, :, :, 1] = i
        right_child_index_mark2[:, :, :, 1] = i


        # assign end point
        left_child_index_mark2[:, :, :, 2] = j
        right_child_index_mark2[:, :, :, 2] = j


        # assign mark
        left_child_index_mark2[:, :, :, 3] = 1
        right_child_index_mark2[:, :, :, 3] = 1

        # assign symbol and index
        left_child_symbol_index_mark2 = symbol_index_t
        left_child_index_mark2[:, :, :, 4:] = left_child_symbol_index_mark2
        right_child_index_mark2[:, :, :, 4:] = left_child_symbol_index_mark2


        self.left_child[i, j, 2] = left_child_index_mark2
        self.right_child[i, j, 2] = right_child_index_mark2
        self.left_child[i, j, 1] = left_child_index_mark1
        self.right_child[i, j, 1] = right_child_index_mark1
