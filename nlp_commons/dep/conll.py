# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# conll.py: Classes to read CoNLL 2006 and 2007 corpora.
# TODO: test projectiveness and project the dependency trees.

import nltk
from nltk.corpus.reader import dependency
from nltk import tree
from nltk import corpus

from dep import depgraph
from dep import depset
import treebank

class CoNLLTreebank(treebank.Treebank):
    def __init__(self, corpus, files=None, max_length=None):
        treebank.Treebank.__init__(self)
        self.corpus = corpus
        self.trees = []
        #print is_punctuation
        i = 0
        non_projectable, empty = 0, 0
        non_leaf = []
        for d in self.corpus.parsed_sents(files):
            # print "Voy por la ", i
            d2 = depgraph.DepGraph(d)
            try:
                d2.remove_leaves(type(self).is_punctuation)
                t = d2.constree()
            except Exception as e:
                msg = e[0]
                if msg.startswith('Non-projectable'):
                    non_projectable += 1
                else:
                    non_leaf += [i]
            else:
                s = t.leaves()
                if s != [] and (max_length is None or len(s) <= max_length):
                    t.corpus_index = i
                    t.depset = depset.from_depgraph(d2)
                    self.trees += [t]
                else:
                    empty += 1
            i += 1
        self.non_projectable = non_projectable
        self.empty = empty
        self.non_leaf = non_leaf
    
    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return False


class CoNLL06Treebank(CoNLLTreebank):
    def __init__(self, root, max_length=None, files=None):
        if files is None:
            files = self.files
        corpus = dependency.DependencyCorpusReader(nltk.data.find('corpora/conll06/data/'+root), files)
        CoNLLTreebank.__init__(self, corpus, None, max_length)


class German(CoNLL06Treebank):
    root = 'german/tiger/'
    files = ['train/german_tiger_train.conll', \
                'test/german_tiger_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'][0] == '$'


class Turkish(CoNLL06Treebank):
    root = 'turkish/metu_sabanci/'
    files = ['train/turkish_metu_sabanci_train.conll', \
                'test/turkish_metu_sabanci_test.conll']
    
    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'Punc'


class Danish(CoNLL06Treebank):
    root = 'danish/ddt/'
    files = ['train/danish_ddt_train.conll', 'test/danish_ddt_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'XP'


class Swedish(CoNLL06Treebank):
    root = 'swedish/talbanken05/'
    files = ['train/swedish_talbanken05_train.conll', 'test/swedish_talbanken05_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'IP'


class Portuguese(CoNLL06Treebank):
    root = 'portuguese/bosque/'
    files = ['treebank/portuguese_bosque_train.conll', 'test/portuguese_bosque_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'punc'


class Arabic(CoNLL06Treebank):
    root = 'arabic/PADT/'
    files = ['train/arabic.train', 'treebank/arabic_PADT_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'G'


class Bulgarian(CoNLL06Treebank):
    root = 'bulgarian/bultreebank/'
    files = ['train/bulgarian_bultreebank_train.conll', 'test/bulgarian_bultreebank_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'Punct'


class Chinese(CoNLL06Treebank):
    root = 'chinese/sinica/'
    files = ['train/chinese_sinica_train.conll', 'test/chinese_sinica_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return False


class Czech(CoNLL06Treebank):
    root = 'czech/pdt/'
    files = ['train/czech.train', 'treebank/czech_pdt_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # see http://ufal.mff.cuni.cz/pdt2.0/doc/manuals/en/m-layer/html/ch02s02s01.html
        # n['tag'] is the fifth column.
        return n['tag'] == ':'


class Dutch(CoNLL06Treebank):
    root = 'dutch/alpino/'
    files = ['train/dutch_alpino_train.conll', 'test/dutch_alpino_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'Punc'


class Japanese(CoNLL06Treebank):
    root = 'japanese/verbmobil/'
    files = ['train/japanese_verbmobil_train.conll', 'test/japanese_verbmobil_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == '.'


class Slovene(CoNLL06Treebank):
    root = 'slovene/sdt/'
    files = ['treebank/slovene_sdt_train.conll', 'test/slovene_sdt_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'] == 'PUNC'


class Spanish(CoNLL06Treebank):
    root = 'spanish/cast3lb/'
    files = ['train/spanish_cast3lb_train.conll', 'test/spanish_cast3lb_test.conll']

    def __init__(self, max_length=None, files=None):
        CoNLL06Treebank.__init__(self, self.root, max_length, files)

    @staticmethod
    def is_punctuation(n):
        # n['tag'] is the fifth column.
        return n['tag'][0] == 'F'


class Catalan(CoNLLTreebank):
    def __init__(self):
        CoNLLTreebank.__init__(self, corpus.conll2007, ['cat.test', 'cat.train'])

    @staticmethod
    def is_punctuation(n):
        return n['tag'].lower()[0] == 'f'


class Basque(CoNLLTreebank):
    def __init__(self):
        CoNLLTreebank.__init__(self, corpus.conll2007, ['eus.test', 'eus.train'])

    @staticmethod
    def is_punctuation(n):
        return n['tag'] == 'PUNT'


def stats():
    cls = [German, Turkish, Danish, Swedish, Portuguese, Arabic, Bulgarian, \
            Chinese, Czech, Dutch, Japanese, Slovene, Spanish]
    for c in cls:
        tb = c(max_length=10)
        print c, len(tb.trees)
