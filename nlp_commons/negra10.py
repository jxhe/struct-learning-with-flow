# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

import itertools

from nltk.util import LazyMap

from . import negra


class Negran(negra.Negra):
    
    def __init__(self, n, basedir=None, load=True):
        negra.Negra.__init__(self, basedir)
        self.n = n
        self.filename = 'negra%02i.treebank' % n
        if load:
            self.get_trees()
    
    def _generate_trees(self):
        print("Parsing treebank...")
        # algunas frases quedan de largo 0 porque son solo '.'
        g = lambda l: (l <= self.n) and (l > 0)
        #f = lambda t: (len(t.leaves()) <= self.n) and (len(t.leaves()) > 0)
        f = lambda t: g(len(t.leaves()))
        m = lambda t: self._prepare(t)
        trees = [t for t in filter(f, map(m, self.parsed()))]
        return trees
    
    def _prepare(self, t):
        t.remove_leaves()
        t.remove_ellipsis()
        t.remove_punctuation()
        return t

    def tagged_sents(self):
        # LazyMap from nltk.util:
        f = lambda t: [(x,x) for x in t.leaves()]
        return LazyMap(f,  self.get_trees())
    
    # XXX: este simplify tags deja un tag '' en el corpus. leer implementacion.
    def simplify_tags(self):
        # XXX: esto no funciona cuando el '-' no esta, leer docs:
        #f = lambda s: s.rpartition('-')[0]
        
        # partition with firt or last '-'?
        # sent 1461 (4391 in the whole corpus) 
        # has tag '--': with 1st: '', with 2nd: '-'.
        # this is the only sentence that has a tag with two '-'s.
        f = lambda s: s.partition('-')[0]
        list(map(lambda t: t.map_leaves(f), self.trees))
        
        # manually fix tree 1461:
        self.trees[1461][1] = '-'


class Negra10(Negran):
    
    def __init__(self, basedir=None, load=True):
        Negran.__init__(self, 10, basedir, load)

"""
Punctuation in NEGRA has some problems:
1. Opening and closing quotes are not distinguished (they are always ").
2. Closing parenthesis are always tagged as opening ('($*LRB* *RRB*)').
3. Quotes are always tagged $*LRB*, getting confused with parenthesis.
4. Single quotes are always tagged $*LRB* ($*LRB*-PNC in two cases). They aren't even real punctuation but possesives in the most cases.
5. What about dashes? slash (/)? elipsis (...)?
5a. Slashes (/) are tagged $*LRB*. They usually are punctuation.
5b. Dashes (-) are tagged $*LRB*. They usually are punctuation. Sometimes bracket punctuation. Some dashes are tagged starting with *. Those are not punctuation but empty elements.
5c. Ellipsis (...) are tagged $*LRB*. They usually are punctuation.
6. In Penn format some punctuation has been crossed. What originally was
    " La mujer de Benjamen " ( Benjamins Frau , 1990 )
became
    " La mujer de Benjamen ( " Benjamins Frau , 1990 )
in Penn format (sentence 120).

In this class we fix problems 2, 3, 4 and 5:
a. To be sure that they are different, parenthesis are tagged $( and )$.
b. Quotes are tagged with $", slashes $/, dashes $d (- is used to separate tag from function), ellipsis '$...'.
c. Single quotes are removed by being tagged as ellipsis '*' (not '...' but empty elements).
"""
class Negra10P(Negra10):
    # sadly I need this list (redundant beacuse we have is_punctuation) to allow 
    # usage by other classes that want to pickle this information:
    # XXX: only sure it works for subclass Negra10:
    punctuation_tags = ['$.', '$/', '$,', '$d', '$.-NMC', '$(', '$)', '$.-CD', '$"', '$...']
    # this was found this way:
    #from negra10 import *
    #tb = Negra10P()
    #punct = set(sum(([x for x in t.leaves() if x[0] == '$'] for t in tb.trees), []))
    
    stop_punctuation_tags = ['$.', '$/', '$,', '$d', '$.-NMC', '$.-CD', '$...']
    bracket_punctuation_tag_pairs = [('$(', '$)'), ('$"',)]
    
    
    def __init__(self, basedir=None, load=True):
        Negra10.__init__(self, basedir, load=False)
        self.filename = 'negra%02ip.treebank' % self.n
        if load:
            self.get_trees()
    
    
    def _generate_trees(self):
        print("Parsing treebank...")
        # algunas frases quedan de largo 0 porque son solo '.'
        g = lambda l: (l <= self.n) and (l > 0)
        #f = lambda t: g(len(filter(lambda x: x not in self.punctuation_tags, t.leaves())))
        f = lambda t: g(len([x for x in t.leaves() if not negra.is_punctuation(x)]))
        m = lambda t: self._prepare(t)
        trees = [t for t in filter(f, map(m, self.parsed()))]
        return trees
    
    
    def _prepare(self, t):
        # before removing leaves, punctuation tags must be fixed:
        for p in t.treepositions('leaves'):
            l = t[p]
            tag = t[p[:-1]]
            if l == '*RRB*':
                tag.node = '$)'
            elif l == '*LRB*':
                tag.node = '$('
            elif l == '"':
                tag.node = '$"'
            elif l == '/':
                tag.node = '$/'
            elif l == '-' and tag.node[0] == '$':
                # some dashes are not punctutation but empty elements 
                # (tag starting with *).
                tag.node = '$d'
            elif l == '...':
                tag.node = '$...'
            elif l == '\'':
                tag.node = '*'
        
        t.remove_leaves()
        t.remove_ellipsis()
        #t.remove_punctuation()
        return t


def test():
    tb = Negra10()
    tb.simplify_tags()
    return tb
