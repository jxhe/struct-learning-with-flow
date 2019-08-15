# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

import itertools

from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

from . import treebank


def is_ellipsis(s):
    #return s[:2] == '*T'
    return s[0] == '*'


def is_punctuation(s):
    return s[0] == '$'


class NegraTree(treebank.Tree):
    
    
    def is_ellipsis(self, s):
        return is_ellipsis(s)
    
    
    def is_punctuation(self, s):
        return is_punctuation(s)


class Negra(treebank.SavedTreebank):
    default_basedir = 'negra-corpus'
    trees = []
    filename = 'negra.treebank'
    
    
    def __init__(self, basedir=None):
        if basedir == None:
            basedir = self.default_basedir
        self.basedir = basedir
        self.reader = BracketParseCorpusReader(basedir, 'negra-corpus2.penn', comment_char='%')
    
    
    def parsed(self, files=None):
        #for t in treebank.SavedTreebank.parsed(self, files):
        for (i, t) in zip(itertools.count(), self.reader.parsed_sents()):
            yield NegraTree(t, labels=i)
    
    
    def get_tree(self, offset=0):
        t = self.get_trees2(offset, offset+1)[0]
        return t
    
    
    # Devuelve los arboles que se encuentran en la posicion i con start <= i < end
    def get_trees2(self, start=0, end=None):
        lt = [t for t in itertools.islice(self.parsed(), start, end)]
        return lt
    
    
    def is_ellipsis(self, s):
        return is_ellipsis(s)
    
    
    def is_punctuation(self, s):
        return is_punctuation(s)


def test():
    tb = Negra()
    trees = tb.get_trees()
    return tb

"""
PREPROCESAMIENTO DEL NEGRA:
    
>>> f = open('negra-corpus/negra-corpus.penn')
>>> g = open('negra-corpus/negra-corpus2.penn', 'w')
>>> for l in f:
...     if l[0] == '(':
...             l = '(ROOT'+l[1:]
...     g.write(l)
...
>>> f.close()
>>> g.close()
"""
