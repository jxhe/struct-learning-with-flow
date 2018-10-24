from __future__ import print_function

import random
import math

from nltk import tree
from utils import stable_log


harmonic_constant = 2.0

def add(dict, x, val):
    dict[x] = dict.get(x, 0) + val

class DMVDict(object):
    def __init__(self, d=None, default_val=math.log(0.1)):
        if d is None:
            self.d = {}
        else:
            self.d = d
        self.default_val = default_val

    def set_default_val(self, val):
        self.default_val = val

    def val(self, x):
        return self.d.get(x, self.default_val)

    def setVal(self, x, val):
        self.d[x] = val

    def add1(self, x):
        self.add(x, 1.0)

    def add(self, x, y):
        add(self.d, x, y)

    def iteritems(self):
        return self.d.items()


def lplace_smooth(tita, count, tag_set, end_symbol, smth_const):
    for h in tag_set:
        tita.add(('attach_left', h, end_symbol), smth_const)
        count.add(('attach_left', end_symbol), smth_const)
        for a in tag_set:
            tita.add(('attach_left', a, h), smth_const)
            tita.add(('attach_right', a, h), smth_const)
            count.add(('attach_right', h), smth_const)
            count.add(('attach_left', h), smth_const)

class DMV(object):
    def __init__(self, args):

        self.end_symbol = 'END'
        self.tita = None

        self.harmonic = False
        self.args = args

    def set_harmonic(self, val):
        self.harmonic = val

    def init_params(self, train_tags, tag_set):
        tita, count = DMVDict(), DMVDict()
        # harmonic initializer
        lplace_smooth(tita, count, tag_set, 
            self.end_symbol, self.args.smth_const)
        self.set_harmonic(True)
        for i, s in enumerate(filter(lambda s: len(s) > 1, \
                                    train_tags)):
            if i % 1000 == 0:
                print('initialize, sentence %d' % i)
            parse_tree, prob = self.dep_parse(s)
            self.MStep_s(parse_tree, tita, count)
        self.MStep(tita, count)

    @staticmethod
    def tree_to_depset(t):
        # add the root symbol (-1)
        res = set([(t.label().index, -1)])
        res.update(DMV._tree_to_depset(t))
        return sorted(res)

    @staticmethod
    def _tree_to_depset(t):
        node = t.label()
        index = node.index
        mark = node.mark
        #res = set([(index, -1)])
        # len(t) is the number of children
        if len(t) > 1:
            if mark == '<>':
                arg = t[0]
            elif mark == '>':
                arg = t[1]
            res = set([(arg.label().index, index)])
            res.update(DMV._tree_to_depset(t[0]), DMV._tree_to_depset(t[1]))
        else:
            if not isinstance(t[0], str):
                res = DMV._tree_to_depset(t[0])
            else:
                res = set()
        return res

    def eval(self, gold, tags, all_len=False):
        """
        Args:
            gold: A nested list of heads
            all_len: True if evaluating on all lengths

        """

        # parse: a list of DepSets
        parse = []
        for k, s in enumerate(tags):
            parse.append(self.tree_to_depset(self.parse(s)))
            if all_len:
                if k % 10 == 0:
                    print('parse %d trees' % k)
        
        cnt = 0
        dir_cnt = 0.0
        undir_cnt = 0.0

        for gold_s, parse_s in zip(gold, parse):
            length = len(gold_s)
            if length > 1:
                (directed, undirected) = self.measures(gold_s, parse_s)
                cnt += length
                dir_cnt += directed
                undir_cnt += undirected

        dir_acu = dir_cnt / cnt
        undir_acu = undir_cnt / cnt

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

    def EStep(self, s):
        pio = self.p_inside_outside(s)

        return pio

    def MStep(self, tita, count):

        for x, p in tita.iteritems():
            p = float(p)
            if p == 0.0:
                raise ValueError
            elif x[0] == 'stop_left':
                tita.setVal(x, math.log(p / count.val(x)))
            elif x[0] == 'stop_right':
                tita.setVal(x, math.log(p / count.val(x)))
            elif x[0] == 'attach_left':
                tita.setVal(x, math.log(p / count.val(('attach_left', x[2]))))
            elif x[0] == 'attach_right':
                tita.setVal(x, math.log(p / count.val(('attach_right', x[2]))))
            p_new = tita.val(x)

            if p_new > 0:
                self.count = count
                print('(x, p, p_new) =', (x, p, p_new))
                raise ValueError
        self.tita = tita

    def _calc_maxval(self, t):
        max_val = 0
        node = t.label()
        max_val = max(node.r_val, node.l_val)

        max_val_list = [max_val]
        for child in t:
            if not isinstance(child, str):
                max_val_list += [self._calc_maxval(child)]

        return max(max_val_list)



    def _calc_stats(self, t, tita, count):
        node = t.label()
        index = node.index
        mark = node.mark
        word = node.word
        l_val = node.l_val
        r_val = node.r_val

        # calc stop denom
        if mark == '>':
            count.add(('stop_right', word, r_val == 0), 1)
        elif mark == '<>':
            count.add(('stop_left', word, l_val == 0), 1)


        if len(t) > 1:
            if mark == '<>':
                arg = t[0]
                tita.add(('attach_left', arg.label().word, word), 1)
                count.add(('attach_left', word), 1)
            elif mark == '>':
                arg = t[1]
                tita.add(('attach_right', arg.label().word, word), 1)
                count.add(('attach_right', word), 1)
            self._calc_stats(t[0], tita, count)
            self._calc_stats(t[1], tita, count)
        else:
            if not isinstance(t[0], str):
                if mark == '|':
                    tita.add(('stop_left', word, l_val == 0), 1)

                elif mark == '<>':
                    tita.add(('stop_right', word, r_val == 0), 1)
                self._calc_stats(t[0], tita, count)
            else:
                assert mark == '>'

    def MStep_s(self, t, tita, count):

        h = self.end_symbol
        count.add(('attach_left', h), 1)
        tita.add(('attach_left', t.label().word, h), 1)
        self._calc_stats(t, tita, count)


    def parse(self, s):
        t, w = self.dep_parse(s)
        return t

    def dep_parse(self, s):
        """
        output:
            returned t is a nltk.tree.Tree without root node
        """
        parse = {}
        # OPTIMIZATION: END considered only explicitly
        # s = s + [self.end_symbol]

        n = len(s)

        for i in range(n):
            j = i + 1
            w = str(s[i])
            t1 = tree.Tree(Node('>', w, i, 0, 0), [w])

            parse[i, j] = ParseDict(self.unary_parses(math.log(1.0), t1, i, j))

        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i + l
                parse_dict = ParseDict()
                for k in range(i+1, j):
                    for (p1, t1) in parse[i, k].itervalues():
                        for (p2, t2) in parse[k, j].itervalues():
                            n1 = t1.label()
                            n2 = t2.label()
                            if n1.mark == '>' and n2.mark == '|':
                                m = n1.index
                                h = n1.word
                                p = self.p_nonstop_right(h, n1.r_val, self.harmonic) + \
                                    self.p_attach_right(n2.word, h, self.harmonic, n2.index - m) + \
                                    p1 + p2
                                new_node = Node(n1.mark, n1.word, n1.index, n1.l_val, n1.r_val + 1)
                                t = tree.Tree(new_node, [t1, t2])
                                parse_dict.add(p, t)
                            if n1.mark == '|' and n2.mark == '<>':
                                m = n2.index
                                h = n2.word
                                p = self.p_nonstop_left(h, n2.l_val, self.harmonic) + \
                                    self.p_attach_left(n1.word, h, self.harmonic, m - n1.index) + \
                                    p1 + p2
                                new_node = Node(n2.mark, n2.word, n2.index, n2.l_val + 1, n2.r_val)
                                t = tree.Tree(new_node, [t1, t2])
                                parse_dict.add(p, t)

                parse[i, j] = ParseDict(sum((self.unary_parses(p, t, i, j) \
                                    for (p, t) in parse_dict.itervalues()), []))

        w = s[0]
        (p1, t1) = parse[0, n].val('|'+w+'0')
        t_max, p_max = t1, p1 + self.p_attach_left(w, self.end_symbol, self.harmonic)
        l = [(t_max, p_max)]
        for i in range(1, n):
            w = s[i]
            (p1, t1) = parse[0, n].val('|'+w+str(i))
            p = p1 + self.p_attach_left(w, self.end_symbol, self.harmonic)
            if p > p_max:
                p_max = p
                l = [(t1, p)]
            elif p == p_max:
                l += [(t1, p)]
        (t_max, p_max) = self.choice(l, self.args.choice)

        return (t_max, p_max)

    def choice(self, l, method):
        """
        select on parse tree from list l,
        which is a list of tuple (t, p)

        """
        if method == 'random':
            return random.choice(l)
        elif method == 'minival':
            (t_min, p_min) = l[0]
            val_min = 10
            for (t, p) in l:
                val = self._calc_maxval(t)
                print(val)
                if val < val_min:
                    (t_min, p_min) = (t, p)
                    val_min = val

            return (t_min, p_min)
        elif method == 'bias_middle':
            (t_min, p_min) = l[0]
            min_dist = 10
            for (t, p) in l:
                middle = (len(t.leaves())) / 2.0
                dist = abs(t.label().index - middle)
                if dist < min_dist:
                    (t_min, p_min) = (t, p)
                    min_dist = dist
            return (t_min, p_min)
        elif method == 'soft_bias_middle':
            new_list = [random.choice(l)]
            for (t, p) in l:
                middle = (len(t.leaves())) / 2.0
                dist = abs(t.label().index - middle) + 1
                if dist < middle:
                    new_list += [(t, p)]
            return random.choice(new_list)
        elif method == 'exclude_end':
            new_list = []
            for (t, p) in l:
                length = len(t.leaves())
                if (t.label().index != 0 and t.label().index != length - 1) or (len(t.leaves()) < 5):
                    new_list += [(t, p)]

            if len(new_list) == 0:
                new_list = l
            return random.choice(new_list)
        elif method == 'bias_left':
            return l[0]



    def unary_parses(self, p, t, i, j):
        node = t.label()
        l_val = node.l_val
        r_val = node.r_val
        if node.mark == '|':
            res = []

        elif node.mark == '<>':
            p2 = self.p_stop_left(node.word, l_val, self.harmonic) + p
            t2 = tree.Tree(Node('|', node.word, node.index, l_val, r_val), [t])
            res = [(p2, t2)]
        elif node.mark == '>':
            p2 = self.p_stop_right(node.word, r_val, self.harmonic) + p
            t2 = tree.Tree(Node('<>', node.word, node.index, l_val, r_val), [t])
            res = self.unary_parses(p2, t2, i, j)
        return [(p, t)] + res

    def p_nonstop_left(self, w, val, harmonic=False):
        try:
            return stable_log(1.0 - math.exp(self.p_stop_left(w, val, harmonic)))
        except ValueError:
            print(math.exp(self.p_stop_left(w, val, harmonic)), 
                self.p_stop_left(w, val, harmonic))

    def p_nonstop_right(self, w, val, harmonic=False):
        return stable_log(1.0 - math.exp(self.p_stop_right(w, val, harmonic)))

    def p_stop_left(self, w, val, harmonic=False):
        if harmonic:
            if val == 0:
                return math.log(self.args.stop_adj)
            else:
                return math.log(1 - self.args.stop_adj)

        return self.tita.val(('stop_left', w, val == 0))

    def p_stop_right(self, w, val, harmonic=False):
        if harmonic:
            if val == 0:
                return math.log(self.args.stop_adj)
            else:
                return math.log(1 - self.args.stop_adj)

        return self.tita.val(('stop_right', w, val == 0))

    def p_attach_left(self, a, h, harmonic=False, dist=None):
        if harmonic:
            if h == self.end_symbol:
                return math.log(0.02)
            return math.log(1.0 / (dist + harmonic_constant))
        return self.tita.val(('attach_left', a, h))

    def p_attach_right(self, a, h, harmonic=False, dist=None):
        if harmonic:
            return math.log(1.0 / (dist + harmonic_constant))
        return self.tita.val(('attach_right', a, h))


class Node(object):
    def __init__(self, mark, word, index, l_val, r_val):
        self.mark = mark
        self.word = word
        self.index = index
        self.l_val = l_val
        self.r_val = r_val

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.mark, self.word, self.index, self.l_val, self.r_val) \
                == (other.mark, other.word, other.index, other.l_val, other.r_val)

    def __str__(self):
        return str(self.mark) + str(self.word) + str(self.index)

    def __repr__(self):
        return self.__str__()


class ParseDict(object):
    def __init__(self, parses=None):
        self.dict = {}
        if parses is not None:
            self.add_all(parses)

    def val(self, node):
        return self.dict[str(node)]

    def add(self, p, t):
        n = t.label()
        s = str(n)
        if (s not in self.dict) or (self.dict[s][0] < p):
            self.dict[s] = (p, t)

    def add_all(self, parses):
        for (p, t) in parses:
            self.add(p, t)

    def itervalues(self):
        return self.dict.values()
