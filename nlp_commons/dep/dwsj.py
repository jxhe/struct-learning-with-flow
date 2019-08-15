# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# dwsj.py: Dependency version of the WSJ corpus.

"""
from dep.dwsj import *
import wsj
tb = wsj.WSJ()
t = tb.get_tree(2)
t.leaves()
# ['Rudolph', 'Agnew', ',', '55', 'years', 'old', 'and', 'former', 'chairman', 'of', 'Consolidated', 'Gold', 'Fields', 'PLC', ',', 'was', 'named', '*-1', 'a', 'nonexecutive', 'director', 'of', 'this', 'British', 'industrial', 'conglomerate', '.']
find_heads(t)
t.depset = tree_to_depset(t)
t.depset.deps
"""

from .. import wsj
from .. import wsj10
from . import depset


class DepWSJ(wsj10.WSJn):

    def __init__(self, max_length, basedir=None, load=True, extra_tags=None):
        wsj10.WSJn.__init__(self, max_length, basedir, load=False, extra_tags=extra_tags)
        self.filename = '%s.treebank' % basedir
        if load:
            self.get_trees()

    def _generate_trees(self):
        trees = wsj10.WSJn._generate_trees(self)

        for t in trees:
            # First find the head for each constituent:
            find_heads(t)
            t.depset = tree_to_depset(t)
        return trees

    def get_gold_dep(self):
        """
        Return:
            gold_dep: a list of DepSet

        """
        gold_dep = [t.depset for t in self.trees]

        return gold_dep




class DepWSJ10(DepWSJ):

    def __init__(self, basedir=None, load=True, extra_tags=None):
        DepWSJ.__init__(self, 10, basedir, load, extra_tags)


def find_heads(t):
    """Mark heads in the constituent tree t using the Collins PhD Thesis (1999)
    rules. The heads are marked in every subtree st in the attributes st.node
    and st.head.
    """
    for st in t.subtrees():
        label = st.label().split('-')[0].split('=')[0]
        # the children may be a tree or a leaf (type string):
        children = [(type(x) is bytes and x) or (type(x) is str and x) or
                    x.label().split('-')[0] for x in st]
        st.head = get_head(label, children)-1
        st.set_label('['+children[st.head]+']')


def tree_to_depset(t):
    """Returns the DepSet associated to the head marked tree t (with find_heads).
    """
    leave_index = 0
    res = set()
    aux = {}
    # Traverse the tree from the leaves upwards (postorder)
    for p in t.treepositions(order='postorder'):
        st = t[p]
        if isinstance(st, bytes) or isinstance(st, str):
            # We are at leave with index leave_index.
            aux[p] = leave_index
            leave_index += 1
        else:
            # We are at a subtree. aux has the index of the
            # head for each subsubtree.
            head = st.head
            if type(st[head]) is bytes or type(st[head]) is str:
                # index of the leave at deptree[head]
                head_index = aux[p+(head,)]
            else:
                head_index = st[head].head_index
            st.head_index = head_index
            for i in range(len(st)):
                sst = st[i]
                if i == head:
                    pass # skip self dependency
                elif type(sst) is bytes or type(sst) is str:
                    res.add((aux[p+(i,)], head_index))
                else:
                    res.add((sst.head_index, head_index))
    res.add((t.head_index, -1))

    return depset.DepSet(len(t.leaves()), sorted(res))


def get_head(label, children):
    """children must be a not empty list. Returns the index of the head,
    starting from 1.
    (rules for the Penn Treebank, taken from p. 239 of Collins thesis).
    The rules for X and NX are not specified by Collins. We use the ones
    at <paste link here> (also at Yamada and Matsumoto 2003).
    (X only appears at wsj_0056.mrg and at wsj_0077.mrg)
    """
    assert children != []
    if len(children) == 1:
        # Used also when label is a POS tag and children is a word.
        res = 1
    elif label == 'NP':
        # Rules for NPs

        # search* returns indexes starting from 1
        # (to avoid confusion between 0 and False):
        res = (children[-1] in wsj.word_tags and len(children)) or \
                searchr(children, set('NN NNP NNS NNPS NNS NX POS JJR'.split())) or \
                searchl(children, 'NP') or \
                searchr(children, set('$ ADJP PRN'.split())) or \
                searchr(children, 'CD') or \
                searchr(children, set('JJ JJS RB QP'.split())) or \
                len(children)
    else:
        rule = head_rules[label]
        plist = rule[1]
        if plist == [] and rule[0] == 'r':
            res = len(children)
        # Redundant:
        #elif plist == [] and rule[0] == 'l':
        #    res = 1
        else:
            res = None
        i, n = 0, len(plist)
        if rule[0] == 'l':
            while i < n and res is None:
                # search* returns indexes starting from 1
                res = searchl(children, plist[i])
                i += 1
        else:
            #assert rule[0] == 'r'
            while i < n and res is None:
                # search* returns indexes starting from 1
                res = searchr(children, plist[i])
                i += 1
        if res is None:
            res = 1

    # Rules for coordinated phrases
    #if 'CC' in [res-2 >= 0 and children[res-2], \
    #            res < len(children) and children[res]]:
    if res-2 >= 1 and children[res-2] == 'CC':
        # On the other case the head doesn't change.
        res -= 2

    return res


def searchr(l, e):
    """As searchl but from right to left. When not None, returns the index
    starting from 1.
    """
    l = l[::-1]
    r = searchl(l, e)
    if r is None:
        return None
    else:
        return len(l)-r+1


def searchl(l, e):
    """Returns the index of the first occurrence of any member of e in l,
    starting from 1 (just for convenience in the usage, see get_head). Returns
    None if there is no occurrence.
    """
    #print 'searchl('+str(l)+', '+str(e)+')'
    if type(e) is not set:
        e = set([e])
    i, n = 0, len(l)
    while i < n and l[i] not in e:
        i += 1
    if i == n:
        return None
    else:
        #return l[i]
        return i+1


head_rules = {'ADJP': ('l', 'NNS QP NN $ ADVP JJ VBN VBG ADJP JJR NP JJS DT FW RBR RBS SBAR RB'.split()), \
 'ADVP': ('r', 'RB RBR RBS FW ADVP TO CD JJR JJ IN NP JJS NN'.split()), \
 'CONJP': ('r', 'CC RB IN'.split()), \
 'FRAG': ('r', []), \
 'INTJ': ('l', []), \
 'LST': ('r', 'LS :'.split()), \
 'NAC': ('l', 'NN NNS NNP NNPS NP NAC EX $ CD QP PRP VBG JJ JJS JJR ADJP FW'.split()), \
 'PP': ('r', 'IN TO VBG VBN RP FW'.split()), \
 'PRN': ('l', []), \
 'PRT': ('r', 'RP'.split()), \
 'QP': ('l', '$ IN NNS NN JJ RB DT CD NCD QP JJR JJS'.split()), \
 'RRC': ('r', 'VP NP ADVP ADJP PP'.split()), \
 'S': ('l', 'TO IN VP S SBAR ADJP UCP NP'.split()), \
 'SBAR': ('l', 'WHNP WHPP WHADVP WHADJP IN DT S SQ SINV SBAR FRAG'.split()), \
 'SBARQ': ('l', 'SQ S SINV SBARQ FRAG'.split()), \
 'SINV': ('l', 'VBZ VBD VBP VB MD VP S SINV ADJP NP'.split()), \
 'SQ': ('l', 'VBZ VBD VBP VB MD VP SQ'.split()), \
 'UCP': ('r', []), \
 'VP': ('l', 'TO VBD VBN MD VBZ VB VBG VBP VP ADJP NN NNS NP'.split()), \
 'WHADJP': ('l', 'CC WRB JJ ADJP'.split()), \
 'WHADVP': ('r', 'CC WRB'.split()), \
 'WHNP': ('l', 'WDT WP WP$ WHADJP WHPP WHNP'.split()), \
 'WHPP': ('r', 'IN TO FW'.split()), \
 'NX': ('r', 'POS NN NNP NNPS NNS NX JJR CD JJ JJS RB QP NP'.split()), \
 'X': ('r', [])
}
