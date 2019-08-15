# dnegra.py: Dependency trees of the NEGRA corpus.

from nltk import tree

import treebank
from dep import depset

class Negra10(treebank.SavedTreebank):
    default_basedir = 'negra-corpus'
    trees = []
    filename = 'negra10.deptreebank'

    def __init__(self, basedir=None, load=True):
        if basedir == None:
            basedir = self.default_basedir
        self.basedir = basedir
        if load:
            self.get_trees()

    def parsed(self):
        f = open(self.basedir+'/negra-corpus.export')
        self.f = f
        
        # go to first sentece
        s = f.readline()
        while not s.startswith('#BOS'):
            s = f.readline()

        while s != '':
            l = s.split()
            (num, origin) = (int(l[1]), int(l[4]))
            sent = []
            l = f.readline().split()
            while l[0][0] != '#':
                #if l[4] != '0':
                if not l[1].startswith('$'):
                    sent += [l]
                l = f.readline().split()
            
            parse = []
            while l[0] != '#EOS':
                parse += [l]
                l = f.readline().split()

            if len(sent) > 0 and len(sent) <= 10:
                self.sent = sent
                self.parse = parse
                t = build_tree(sent, parse)
                t2 = treebank.Tree(t, (num, origin))
                t2.depset = tree_to_depset(t)
                yield t2
            
            s = f.readline()


def build_tree(sent, parse):
    entries = dict((l[0], l) for l in parse)
    # for sentences that have several roots (e.g. #BOS 77 3 863208763 1):
    entries['#0'] = ['#0', 'ROOT', '--', '--', '']

    # add indexed lexical entries:
    for i in range(len(sent)):
        entries[i] = sent[i]

    return _build_tree(entries, '#0')


def _build_tree(entries, root):
    """Helper for build_tree. (Fue un dolor de huevos.)
    """
    entry = entries[root]
    if isinstance(root, int):
        t = tree.Tree(entry[1], [entry[0]])
        t.head = 0
        t.start_index = root
        t.edge = entry[3]
        return t
    else:
        root = root[1:]
        subtrees = []
        for (word, l) in entries.iteritems():
            # parent = l[4]
            if l[4] == root:
                subtree = _build_tree(entries, word)
                subtrees += [subtree]
        subtrees = sorted(subtrees, key=lambda t: t.start_index)
        t = tree.Tree(entry[1], subtrees)
        t.start_index = subtrees[0].start_index
        t.edge = entry[3]

        # head-finding from http://maltparser.org/userguide.html:
        (dir, plist) = head_rules['CAT:'+t.node]
        plist = plist + [('LEXICAL', '')]
        if dir == 'r':
            # we will reverse again later.
            subtrees.reverse()
        found = False
        i = 0
        while i < len(plist) and not found:
            (type, val) = plist[i]
            j = 0
            while j < len(subtrees) and not found:
                subtree = subtrees[j]
                if (type == 'LABEL' and subtree.edge == val) or \
                        (type == 'CAT' and subtree.node.split('[')[0] == val) or \
                        (type == 'LEXICAL' and isinstance(subtree[0], str)):
                    head_st = subtree
                    found = True
                j += 1
            i += 1
        if not found:
            head_st = subtrees[0]
        if dir == 'r':
            subtrees.reverse()

        #if t.node == 'ROOT':
        #    print dir, plist, subtrees, head_st

        # mark head:
        t.head = subtrees.index(head_st)
        t.node += '['+subtrees[t.head].node.split('[')[0]+']'

        return t


head_rules = \
{'CAT:ROOT': ('l', []), \
 'CAT:AA': ('r', [('LABEL', 'HD')]), \
 'CAT:AP': ('r', [('LABEL', 'HD')]), \
 'CAT:AVP': ('r', [('LABEL', 'HD'), ('CAT', 'AVP')]), \
 'CAT:CAC': ('l', [('LABEL', 'CJ')]), \
 'CAT:CAP': ('l', [('LABEL', 'CJ')]), \
 'CAT:CAVP': ('l', [('LABEL', 'CJ')]), \
 'CAT:CCP': ('l', [('LABEL', 'CJ')]), \
 'CAT:CH': ('l', []), \
 'CAT:CNP': ('l', [('LABEL', 'CJ')]), \
 'CAT:CO': ('l', [('LABEL', 'CJ')]), \
 'CAT:CPP': ('l', [('LABEL', 'CJ')]), \
 'CAT:CS': ('l', [('LABEL', 'CJ')]), \
 'CAT:CVP': ('l', [('LABEL', 'CJ')]), \
 'CAT:CVZ': ('l', [('LABEL', 'CJ')]), \
 'CAT:DL': ('l', [('LABEL', 'DH')]), \
 'CAT:ISU': ('l', []), \
 'CAT:NM': ('r', []), \
 'CAT:NP': ('r', [('LABEL', 'NK')]), \
 # Malt says 'PN' (why?)
 'CAT:MPN': ('l', []), \
 'CAT:PP': ('r', [('LABEL', 'NK')]), \
 'CAT:S': ('r', [('LABEL', 'HD')]), \
 'CAT:VP': ('r', [('LABEL', 'HD')]), \
 'CAT:VROOT': ('l', []), \
 # missing rules:
 # e.g. BOS 507:
 'CAT:VZ': ('l', []), \
 # e.g. BOS 5576:
 'CAT:MTA': ('l', []) \
 }


def tree_to_depset(t):
    """Returns the DepSet associated to the partially head marked tree t.
    """
    (res, head) = _tree_to_depset(t)
    if head != -1:
        res.append((head, -1))
    return depset.DepSet(len(t.leaves()), sorted(res))


def _tree_to_depset(t):
    """Helper for tree_to_depset. (Fue un dolor de huevos.)
    """
    #if isinstance(t, str):
    #    return ([], [])
    if isinstance(t[0], str):
        return ([], t.start_index)
    else:
        depset = []
        heads = []
        for st in t:
            (d, h) = _tree_to_depset(st)
            depset += d
            heads += [h]
        if t.head != -1:
            # resolve all unresolved dependencies:
            new_head = heads[t.head]
            new_depset = [(i, (j==-1 and new_head) or j) for (i, j) in depset]
        else:
            # propagate unresolved dependencies
            new_head = -1
            new_depset = depset
        new_depset += [(j, new_head) for j in heads if j != -1 and j != new_head]
        return (new_depset, new_head)


def build_tree2(sent, parse):
    """Iterative version of build_tree. Maybe faster, but uglyer.
    """
    dparse = dict((l[0][1:], l) for l in parse)
    # for sentences that have several roots (e.g. #BOS 77 3 863208763 1):
    dparse['0'] = ['#0', 'ROOT', '--', '--', '']

    ltree = []
    for i in range(len(sent)):
        l = sent[i]
        t = tree.Tree(l[1], [l[0]])
        t.head = 0
        # to be used in tree_to_depset:
        t.start_index = i
        ltree += [(t, l[3], l[4])]

    # (XXX: not sure about the condition)
    while len(ltree) > 1:
        ids = set(dparse.keys()) - set(l[4] for l in dparse.itervalues())
        new_ltree = []
        last_id = -1
        for (t, edge, id) in ltree:
            if id in ids:
                if last_id != id:
                    new_t = tree.Tree(dparse[id][1], [t])
                    new_t.head = -1
                    new_ltree += [(new_t, dparse[id][3], dparse[id][4])]
                else:
                    new_t = new_ltree[-1][0]
                    new_t.append(t)
                # basic head-finding:
                if edge == 'HD':
                    new_t.head = len(new_t)-1
                    new_t.node += '['+t.node+']'
                """# head-finding from http://maltparser.org/userguide.html:
                # (section "Phrase structure parsing")
                elif hasattr(t, 'start_index') and new_t.head == -1:
                    # "hasattr" says that t is a lexical item.
                    new_t.head = len(new_t) - 1
                    new_t.node += '['+t.node+']'"""
            else:
                new_ltree += [(t, edge, id)]
            last_id = id

        for id in ids:
            del dparse[id]
        ltree = new_ltree

    result = ltree[0][0]
    result.depset = depset

    return result


"""
head_rules = { \
    'CAT:AA':  ('r',       'r[LABEL:HD]'), \
    'CAT:AP':  ('r',       'r[LABEL:HD]'), \
    'CAT:AVP': ('r',       'r[LABEL:HD CAT:AVP]'), \
    'CAT:CAC': ('l',       'l[LABEL:CJ]'), \
    'CAT:CAP': ('l',       'l[LABEL:CJ]'), \
    'CAT:CAVP':('l',       'l[LABEL:CJ]'), \
    'CAT:CH':  ('l',       '*'), \
    'CAT:CNP': ('l',       'l[LABEL:CJ]'), \
    'CAT:CO':  ('l',       'l[LABEL:CJ]'), \
    'CAT:CPP': ('l',       'l[LABEL:CJ]'), \
    'CAT:CS':  ('l',       'l[LABEL:CJ]'), \
    'CAT:CVP': ('l',       'l[LABEL:CJ]'), \
    'CAT:CCP': ('l',       'l[LABEL:CJ]'), \
    'CAT:CVZ': ('l',       'l[LABEL:CJ]'), \
    'CAT:DL':  ('l',       'l[LABEL:DH]'), \
    'CAT:ISU': ('l',       '*'), \
    'CAT:NM':  ('r',       '*'), \
    'CAT:NP':  ('r',       'r[LABEL:NK]'), \
    'CAT:PN':  ('l',       '*'), \
    'CAT:PP':  ('r',       'r[LABEL:NK]'), \
    'CAT:S':   ('r',       'r[LABEL:HD]'), \
    'CAT:VROOT':('l',       '*'), \
    'CAT:VP':  ('r',       'r[LABEL:HD]') }

>>> for k,v in head_rules.iteritems():
...     (dir,plist)=v
...     plist = plist[2:-1].split()
...     plist = [(s.split(':')[0], s.split(':')[1]) for s in plist]
...     head_rules[k] = (dir,plist)
...
>>> from pprint import pprint
>>> pprint(head_rules)
"""
