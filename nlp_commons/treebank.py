# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# -*- coding: iso-8859-1 -*-

import os
import itertools

from nltk import tree
from nltk.corpus.reader import api
from nltk.corpus.reader.api import SyntaxCorpusReader
from nltk.util import LazyMap

from . import util
from functools import reduce


class Tree(tree.Tree):
    def __new__(cls, nltk_tree=None, labels=None):
        if nltk_tree is None:
            return super(Tree, cls).__new__(cls)
        else:
            return super(Tree, cls).__new__(cls, nltk_tree.label(), nltk_tree)

    def __init__(self, nltk_tree, labels=None):
        """
        Args:
            nltk_tree: instance of tree.Tree

        """
        tree.Tree.__init__(self, nltk_tree.label(), nltk_tree)
        #TODO(junxian): to be optimized
        # tree.ParentedTree.__init__(self, nltk_tree.label(),
        #                            [tree.ParentedTree.convert(child) \
        #                             for child in nltk_tree])
        self.labels = labels

    def copy(self, deep=False):
        if not deep:
            return self.__class__(self, self.labels)
        else:
            return self.__class__(tree.Tree.convert(self), self.labels)

    def map_nodes(self, f):
        lpos = self.treepositions()
        for pos in lpos:
            if isinstance(self[pos], tree.Tree):
                self[pos].node = f(self[pos].node)
            else:
                self[pos] = f(self[pos])

    def map_leaves(self, f):
        lpos = self.treepositions('leaves')
        for pos in lpos:
            self[pos] = f(self[pos])

    def filter_subtrees(self, f):
        def recursion(t, f):
            # terminals are pos tags
            # if not isinstance(t, tree.Tree):
            #     return t
            if t.height() == 2:
                return t
            subtrees = []
            for st in t:
                if f(st):
                    st = recursion(st, f)
                    subtrees += [st]
            if subtrees == []:
                # ideally, subtrees cannot be empty
                return t.label()
            else:
                return tree.Tree(t.label(), subtrees)
        t = recursion(self, f)
        if isinstance(t, tree.Tree):
            self.__init__(t, self.labels)
        else:
            self.__init__(tree.Tree(t, []), self.labels)

    def remove_functions(self):
        self.map_nodes(lambda node: node.split('-')[0])

    def remove_leaves(self):
        self.filter_subtrees(lambda t: isinstance(t, tree.Tree))

    """def filter_tags(self, valid_tags):
        def f(t):
            if isinstance(t, tree.Tree):
                all_invalid = True
                for leave in t.leaves():
                    all_invalid = all_invalid and not (leave in valid_tags)
                return not all_invalid
            else:
                return (t in valid_tags)
        self.filter_subtrees(f)"""

    # XXX: esta funcion estaria mejor llamada filter_leaves ya que solo filtra
    # tags si estos estan en las hojas...
    def filter_tags(self, tag_filter):
        """tag_filter must be a predicate function over strings.
        """
        def f(t):
            # t must be a tree, with leaves as words
            if t.height() > 2:
                all_invalid = True
                for pre_terminal in t.subtrees(lambda x: x.height() == 2):
                    all_invalid = all_invalid and not tag_filter(pre_terminal.label())
                return not all_invalid
            elif t.height() == 2:
                return tag_filter(t.label())

            # terminals are pos tags
            # if isinstance(t, tree.Tree):
            #     all_invalid = True
            #     for leave in t.leaves():
            #         all_invalid = all_invalid and not tag_filter(leave)
            #     return not all_invalid
            # else:
            #     return tag_filter(t)
        self.filter_subtrees(f)

    def remove_punctuation(self):
        def f(t):
            if isinstance(t, tree.Tree):
                punctuation = True
                for leave in t.leaves():
                    punctuation = punctuation and self.is_punctuation(leave)
                return not punctuation
            else:
                return not self.is_punctuation(t)
        self.filter_subtrees(f)

    def is_punctuation(self, s):
        """To be overriden in the subclasses.
        """
        return False

    def remove_ellipsis(self):
        def f(t):
            if isinstance(t, tree.Tree):
                ellipsis = True
                for leave in t.leaves():
                    ellipsis = ellipsis and self.is_ellipsis(leave)
                return not ellipsis
            else:
                return not self.is_ellipsis(t)
        self.filter_subtrees(f)

    def is_ellipsis(self, s):
        """To be overriden in the subclasses.
        """
        return False

    # DEPRECATED: creo que nunca uso esta bosta, aunque podria:
    # Invocar solo sobre arboles que tengan la frase en sus hojas.
    # Largo de la frase sin contar puntuacion ni elementos nulos.
    # FIXME: esta funcion deberia estar en una clase WSJ_Tree(Tree) en wsj.py.
    def length(self):
        t2 = self.copy()
        t2.remove_leaves()
        return len([t for t in t2.leaves() if t in self.valid_tags])

    def dfs(self):
        queue = self.treepositions()
        stack = [queue.pop(0)]
        while stack != []:
            p = stack[-1]
            if queue == [] or queue[0][:-1] != p:
                stack.pop()
                print(p, "volviendo")
            else: # queue[0] es hijo de p:
                q = queue.pop(0)
                stack.append(q)
                print(p, "yendo")

    def labelled_spannings(self, leaves=True, root=True, unary=True):
        queue = self.treepositions()
        stack = [(queue.pop(0),0)]
        j = 0
        result = []
        while stack != []:
            (p,i) = stack[-1]
            if queue == [] or queue[0][:-1] != p:
                # ya puedo devolver spanning de p:
                if isinstance(self[p], tree.Tree):
                    result += [(self[p].node, (i, j))]
                else:
                    # es una hoja:
                    if leaves:
                        result += [(self[p], (i, i+1))]
                    j = i+1
                stack.pop()
            else: # queue[0] es el sgte. hijo de p:
                q = queue.pop(0)
                stack.append((q,j))
        if not root:
            # El spanning de la raiz siempre queda al final:
            result = result[0:-1]
        if not unary:
            result = [l_i_j for l_i_j in result if l_i_j[1][0] != l_i_j[1][1]-1]
        return result

    def spannings(self, leaves=True, root=True, unary=True):
        """Returns the set of unlabeled spannings.
        """
        queue = self.treepositions()
        stack = [(queue.pop(0),0)]
        j = 0
        result = set()
        while stack != []:
            (p,i) = stack[-1]
            if queue == [] or queue[0][:-1] != p:
                # ya puedo devolver spanning de p:
                if isinstance(self[p], tree.Tree):
                    result.add((i, j))
                else:
                    # es una hoja:
                    if leaves:
                        result.add((i, i+1))
                    j = i+1
                stack.pop()
            else: # queue[0] es el sgte. hijo de p:
                q = queue.pop(0)
                stack.append((q,j))
        if not root:
            # FIXME: seguramente se puede programar mejor:
            result.remove((0,len(self.leaves())))
        if not unary:
            # FIXME: seguramente se puede programar mejor:
            result = set([x_y for x_y in result if x_y[0] != x_y[1]-1])
        return result

    def spannings2(self, leaves=True, root=True, unary=True, order=None):
        """TODO: Returns the unlabeled spannings as an ordered list.
        Meant to replace spannings in the future.
        """
        queue = self.treepositions(order)
        stack = [(queue.pop(0),0)]
        j = 0
        result = set()
        while stack != []:
            (p,i) = stack[-1]
            if queue == [] or queue[0][:-1] != p:
                # ya puedo devolver spanning de p:
                if isinstance(self[p], tree.Tree):
                    result.add((i, j))
                else:
                    # es una hoja:
                    if leaves:
                        result.add((i, i+1))
                    j = i+1
                stack.pop()
            else: # queue[0] es el sgte. hijo de p:
                q = queue.pop(0)
                stack.append((q,j))
        if not root:
            # FIXME: seguramente se puede programar mejor:
            result.remove((0,len(self.leaves())))
        if not unary:
            # FIXME: seguramente se puede programar mejor:
            result = set([x_y1 for x_y1 in result if x_y1[0] != x_y1[1]-1])
        return result


def measures(gold, parse):
    result = labelled_measures(gold, parse)
    bm = bracketed_measures(gold, parse)
    result.update(bm)
    return result


def labelled_measures(gold, parse):
    gold_spans = gold.labelled_spannings(leaves=False, root=False)
    parse_spans = parse.labelled_spannings(leaves=False, root=False)

    hits, l_hits, cb = 0, 0, 0
    for span in parse_spans:
        n = 0
        # Primero busco coincidencia sin label:
        while n < len(gold_spans) and span[1] != gold_spans[n][1]:
            n += 1
        if n < len(gold_spans):
            # Encontre coincidencia sin label. Busco con label:
            hits += 1
            while n < len(gold_spans) and span != gold_spans[n]:
                n += 1
            if n < len(gold_spans):
                l_hits += 1

        # Vemos si tiene brackets consistentes:
        def consistent(span1, span2):
            (i1, j1) = span1[1]
            (i2, j2) = span2[1]
            j1 -= 1
            j2 -= 1
            # Disjuntos, 1 dentro de 2 o 2 dentro de 1
            return j1 < i2 or j2 < i1 or \
                   (i2 <= i1 and j1 <= j2) or \
                   (i1 <= i2 and j2 <= j1)
        n = 0
        while n < len(gold_spans) and consistent(span, gold_spans[n]):
            n += 1
        if n == len(gold_spans):
            cb += 1

    return {'labelled_precision': float(l_hits) / float(len(parse_spans)),
            'labelled_recall': float(l_hits) / float(len(gold_spans)),
            #'bad_bracketed_precision': float(hits) / float(len(parse_spans)),
            #'bad_bracketed_recall': float(hits) / float(len(gold_spans)),
            #'bad_consistent_brackets_recall': float(cb) / float(len(gold_spans))
            }


def bracketed_measures(gold, parse):
    """Unlabeled measures.
    """
    gold_spans = gold.spannings(leaves=False)
    parse_spans = parse.spannings(leaves=False)

    # XXX: Podria hacer hits = len(s.intersection(t)) (o len(s&t)).
    hits, cb = 0, 0
    for span in parse_spans:
        if span in gold_spans:
            # Encontre coincidencia sin label.
            hits += 1

        # Vemos si tiene brackets consistentes:
        def consistent(span1, span2):
            (i1, j1) = span1
            (i2, j2) = span2
            j1 -= 1
            j2 -= 1
            # Disjuntos, 1 dentro de 2 o 2 dentro de 1
            return j1 < i2 or j2 < i1 or \
                   (i2 <= i1 and j1 <= j2) or \
                   (i1 <= i2 and j2 <= j1)
        # XXX: si aparece algun not consistent puedo terminar.
        n = 0
        # XXX: no me gusta usar break.
        for g_span in gold_spans:
            if consistent(span, g_span):
                n += 1
            else:
                break
        if n == len(gold_spans):
            cb += 1

    return {'bracketed_precision': float(hits) / float(len(parse_spans)),
            'bracketed_recall': float(hits) / float(len(gold_spans)),
            'consistent_brackets_recall': float(cb) / float(len(gold_spans))}


def empty_measures():
    return {'labelled_precision': 0.0,
            'bracketed_precision': 0.0,
            'labelled_recall': 0.0,
            'bracketed_recall': 0.0,
            'consistent_brackets_recall': 0.0,
            #'bad_bracketed_precision': 0.0,
            #'bad_bracketed_recall': 0.0,
            #'bad_consistent_brackets_recall': 0.0
            }

"""def labelled_precision(gold, parse):
    return precision(gold, parse, labelled=True)


def precision(gold, parse, labelled=True):
    gold_spans = gold.spannings(leaves=False, root=False)
    parse_spans = parse.spannings(leaves=False, root=False)
    hits = span_hits_count(gold_spans, parse_spans, labelled)
    return float(hits) / float(len(parse_spans))
def span_hits_count(gold_spans, parse_spans, labelled=False):
    #gold_spans = gold.spannings(leaves=False, root=False)
    #parse_spans = parse.spannings(leaves=False, root=False)
#    for (label, (i,j)) in parse_spans:
    if labelled:
        aux = 0
    else:
        aux = 1
    hits = 0
    for span in parse_spans:
        n = 0
        while n < len(gold_spans) and span[aux:] != gold_spans[n][aux:]:
            n += 1
        if n < len(gold_spans):
            hits += 1
    return hits
"""


class Treebank(SyntaxCorpusReader):
    trees = None

    def __init__(self, trees=None):
        if trees is None:
            trees = []
        self.trees = trees
        # super.__init__(self.basedir)

    def get_trees(self):
        return self.trees

    def sents(self):
        # LazyMap from nltk.util:
        return LazyMap(lambda t: t.leaves(),  self.get_trees())

    def tagged_sents(self):
        # LazyMap from nltk.util:
        return LazyMap(lambda t: t.pos(),  self.get_trees())

    def parsed_sents(self):
        return self.get_trees()

    def sent(self, i):
        return self.trees[i].leaves()

    def remove_functions(self):
        list(map(lambda t: t.remove_functions(), self.trees))

    def remove_leaves(self):
        list(map(lambda t: t.remove_leaves(), self.trees))

    def length_sort(self):
        self.trees.sort(lambda x,y: cmp(len(x.leaves()), len(y.leaves())))

    def stats(self, filename=None):
        trees = self.trees
        avg_height = 0.0
        words = 0
        # vocabulary = []
        if filename is not None:
            f = open(filename, 'w')
        for t in trees:
            height = t.height()
            length = len(t.leaves())
            if filename is not None:
                f.write(str(length) + '\n')
            avg_height = avg_height + height
            words = words + length
        if filename is not None:
            f.close()
        avg_height = avg_height / len(trees)
        avg_length = float(words) / len(trees)
        return (len(trees), avg_height, avg_length)

    def print_stats(self, filename=None):
        (size, height, length) = self.stats(filename)
        #print "Pares arbol oracion:", size
        #print "Altura de arbol promedio:", height
        #print "Largo de oracion promedio:", length
        #print "Vocabulario:", len(self.get_vocabulary())
        print("Trees:", size)
        print("Average tree depth:", height)
        print("Average sentence length:", length)
        print("Vocabulary size:", len(self.get_vocabulary()))

    def get_productions(self):
#        productions = []
#        for t in self.trees:
#            productions += t.productions()
        def concat(l):
            return reduce(lambda x,y: x + y, l)
        productions = concat([t.productions() for t in self.trees])
        return productions

    def get_vocabulary(self):
        """Returns the set of terminals of all the trees.
        """
        result = set()
        for t in self.trees:
            result.update(t.leaves())
        return result

    def word_freqs(self):
        d = {}
        for s in self.sents():
            for w in s:
                if w in d:
                    d[w] += 1
                else:
                    d[w] = 1
        return d

    def length_freqs(self):
        d = {}
        for s in self.sents():
            l = len(s)
            if l in d:
                d[l] += 1
            else:
                d[l] = 1
        return d

    def is_punctuation(self, s):
        """To be overriden in the subclasses.
        """
        return False

    def is_ellipsis(self, s):
        """To be overriden in the subclasses.
        """
        return False

    def find_sent(self, ss):
        """Returns the indexes of the sentences that contains the
        sequence of words ss.
        """
        ss = ' '.join(ss)
        l = []
        for i in range(len(self.trees)):
            s = self.sent(i)
            s = ' '.join(s)
            if ss in s:
                l.append(i)
        return l


EMPTY = Treebank([])


def treebank_from_sentences(S):
    """Returns a treebank with sentences S and trivial trees.
    """
    trees = [Tree(tree.Tree('ROOT', [tree.Tree(x, [x]) for x in s])) for s in S]
    return Treebank(trees)


def load_treebank(filename):
    return util.load_obj(filename)


class SavedTreebank(Treebank):
    trees = []

    def __init__(self, filename, basedir):
        self.filename = filename
        self.basedir = basedir

    def get_trees(self):
        if self.trees == []:
            # attempt to load cache
            # trees = util.load_obj(self.filename)
            trees = None
            if trees is None or trees == []:  # not cached yet
                trees = self._generate_trees()
                # util.save_obj(trees, self.filename) # save cache
            self.trees = trees
        return self.trees

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        util.save_obj(self.trees, filename)

    def _generate_trees(self):
        print("Parsing treebank...")
        trees = [self._prepare(t) for t in self.parsed()]
        return trees

    def _prepare(self, t):
        """To be overriden in the subclasses.
        """
        return t

    def parsed(self, files=None):
        """
        Prepared for Penn format. May be overriden.

        @param files: One or more treebank files to be processed
        @type files: L{string} or L{tuple(string)}
        @rtype: iterator over L{tree}
        """
        if files is None:
            files = os.listdir(self.basedir)

        # Just one file to process?  If so convert to a tuple so we can iterate
        if isinstance(files, str):
            files = (files,)

        for file in files:
            print("Parsing file "+file)
            path = os.path.join(self.basedir, file)
            s = open(path).read()
            # i = 0
            for i,t in zip(itertools.count(), tokenize_paren(s)):
                yield Tree(tree.bracket_parse(t), [file, i])
                # i += 1


def tokenize_paren(s):
    """
    Tokenize the text (separated by parentheses).

    @param s: the string or string iterator to be tokenized
    @type s: C{string} or C{iter(string)}
    @return: An iterator over tokens
    """
    k = 0
    t = ""
    for c in s:
        if k >= 1:
            t = t + c

        if c == '(':
            k = k + 1
        elif c == ')':
            k = k - 1
            if k == 0:
                yield t[:-1]
                t = ""
