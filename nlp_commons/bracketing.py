# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# bracketing.py: Bracketing data structure.

import itertools
import random
import string
import math

from nltk import tree

from . import treebank

class Bracketing:
    """For instance: 
    Bracketing(10, set([(1, 3), (5, 11), (6, 11), (8, 10), (1, 4), (7, 11),
    (4, 11)]), 1).
    """
    
    # FIXME: eliminar brackets unarios.
    def __init__(self, length, brackets=None, start_index=0):
        """brackets debe ser un set de pares de enteros.
        """
        
        self.length = length
        self.start_index = start_index
        if brackets is None:
            self.brackets = set()
        else:
            brackets.discard((start_index, start_index+length))
            self.brackets = brackets
    
    def __eq__(self, other):
        if not isinstance(other, Bracketing):
            return False
        return (self.length, self.brackets, self.start_index) == \
                    (other.length, other.brackets, other.start_index)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __le__(self, other):
        if not isinstance(other, Bracketing): 
            return False
        return (self.length, self.start_index) == \
                    (other.length, other.start_index) and \
                self.brackets <= other.brackets
    
    def has(self, xxx_todo_changeme9):
        """Returns True if the bracket belongs to the bracketing or encloses
        the whole sentence."""
        (i, j) = xxx_todo_changeme9
        return j - i == 1 or \
                (i, j) == (self.start_index, self.start_index+self.length) or \
                (i, j) in self.brackets
    
    def has_opening_bracket(self, i, whole=True):
        if whole and i == b.start_index:
            return True
        bs = [a_b3 for a_b3 in self.brackets if a_b3[0] == i]
        return bs != []
    
    def has_closing_bracket(self, i, whole=True):
        if whole and i == self.start_index + self.length:
            return True
        bs = [a_b4 for a_b4 in self.brackets if a_b4[1] == i]
        return bs != []
    
    def ibrackets(self, whole=False, unary=False):
        """Iterator over the brackets.
        """
        if unary and (whole or self.length > 1):
            c1 = map(lambda a: (a, a+1), list(range(self.start_index, self.start_index+self.length)))
        else:
            c1 = []
        #if whole and self.length > 1:
        if whole:
            c3 = [(self.start_index, self.start_index + self.length)]
        else:
            c3 = []
        
        return itertools.chain(c1, self.brackets, c3)
    
    def set_start_index(self, start_index):
        """Change internal representation.
        """
        old = self.start_index
        new = start_index
        self.brackets = set([(a_b[0] - old + new, a_b[1] - old + new) for a_b in self.brackets])
        self.start_index = new
   
    def is_binary(self):
        return (self.length < 3 or len(self.brackets) == self.length - 2) and \
                self.non_crossing()
    
    def non_crossing(self):
        if len(self.brackets) < 2:
            return True
        
        def consistent(xxx_todo_changeme, xxx_todo_changeme8):
            # Disjuntos, 1 dentro de 2 o 2 dentro de 1
            (i1, j1) = xxx_todo_changeme
            (i2, j2) = xxx_todo_changeme8
            return j1 <= i2 or j2 <= i1 or \
                   (i2 <= i1 and j1 <= j2) or \
                   (i1 <= i2 and j2 <= j1)
        
        result = True
        blist = list(self.brackets)
        i, j, l = 0, 1, len(blist)
        while result and (i, j) != (l-1, l):
            result = result and consistent(blist[i], blist[j])
            if j < l-1:
                j += 1
            else:
                i += 1
                j = i+1
        return result
    
    def treefy(self, s=None):
        if s is None:
            s = ['X'] * self.length
        b2 = set([(a_b1[0]-self.start_index, a_b1[1]-self.start_index) for a_b1 in self.brackets])
        return treefy(s, b2)
    
    def strfy(self, s, whole=False):
        """Returns a string representation of the bracketing, using
        s as the bracketed sentence (e.g. 'DT (VB NN)').
        """
        s2 = [x for x in s]
        for (i, j) in self.ibrackets(whole=whole):
            s2[i] = '('+s2[i]
            s2[j-1] = s2[j-1]+')'
        return string.join(s2)
    
    def randomly_binarize(self, start=None, end=None):
        """Binarize the bracketing adding the missing brackets randomly.
        (start and end are used for the recursive call, do not use.)
        """
        brackets = self.brackets
        if start is None:
            first = True
            l = self.length
            start = self.start_index
            end = start + l
        else:
            first = False
            l = end - start
        
        if l > 2:
            # lo primero es identificar los split points posibles:
            splits = []
            i = 1
            while i < l:
                if self.splittable(start + i, start, end):
                    splits += [i]
                i += 1
            """if first:
                print splits
            else:
                print 'start', start, 'end', end"""
            assert splits != []
            
            # ahora elegimos un split al azar y agregamos los brackets:
            split = start + random.choice(splits)
            # esto elegiria si quiero binarizar lo mas parecido posible a rbranch:
            #split = start + splits[0]
            
            if start + 1 < split:
                brackets.add((start, split))
            if split + 1 < end:
                brackets.add((split, end))
            
            # ahora llenamos adentro
            self.randomly_binarize(start=start, end=split)
            self.randomly_binarize(start=split, end=end)
   
    def splittable(self, x, start=None, end=None):
        """Helper for randomly_binarize.
        """
        if start is None:
            start = self.start_index
        if end is None:
            end = self.length
        bs = [a_b5 for a_b5 in list(self.brackets) if start < a_b5[0] or a_b5[1] < end]
        i = 0
        while i < len(bs) and (bs[i][1] <= x or x <= bs[i][0]):
            i += 1
        if i == len(bs):
            return True
        else:
            return False
    
    def reverse(self):
        """Reverse the bracketing.
        """
        s = self.start_index
        n = self.length
        self.brackets = set((n-j+2*s, n-i+2*s) for (i, j) in self.brackets)


def coincidences(b1, b2):
    """Count coincidences between two bracketings.
    """
    s1 = set([(x_y[0] - b1.start_index, x_y[1] - b1.start_index) for x_y in b1.brackets])
    s2 = set([(x_y6[0] - b2.start_index, x_y6[1] - b2.start_index) for x_y6 in b2.brackets])
    return len(s1 & s2)


def treefy(s, b):
    """Convert a binary bracketing b of a sentence s to a NLTK tree.
        b is a set and must not have the trivial top bracket.
    """
    l = len(s)
    if l == 2:
        t = tree.Tree('X', [s[0], s[1]])
    # buscar los hijos de la raiz:
    elif (0, l-1) in b:
        b2 = b - set((0, l-1))

        t2 = treefy(s[:-1], b2)

        t = tree.Tree('X', [t2, s[-1]])
    elif (1, l) in b:
        b2 = b - set((1, l))
        b2 = set([(i_j[0]-1, i_j[1]-1) for i_j in b2])

        t2 = treefy(s[1:], b2)

        t = tree.Tree('X', [s[0], t2])
    else:
        x = 2
        while not ((0, x) in b and (x, l) in b):
            x = x + 1

        b2 = set((i, j) for (i, j) in b if 0 <= i and j <= x)
        b3 = set((i-x, j-x) for (i, j) in b if x <= i and j <= l)
        
        t2 = treefy(s[:x], b2)
        t3 = treefy(s[x:], b3)
        
        t = tree.Tree('X', [t2, t3])

    return t


def string_to_bracketing(s):
    """Converts a string to a bracketing.

    >>> string_to_bracketing('(DT NNP NN) (VBD (DT (VBZ (DT JJ NN))))')
    """
    s2 = s.replace('(', '(X ')
    s2 = '((X '+s2+'))'
    t = treebank.Tree(tree.bracket_parse(s2))
    b = tree_to_bracketing(t)
    return b


def tree_to_bracketing(t, start_index=0):
    """t must be instance of treebank.Tree.
    """
    l = len(t.leaves())
    spans = t.spannings(leaves=False,root=False,unary=False)
    moved_spans = set([(a_b7[0]+start_index, a_b7[1]+start_index) for a_b7 in spans])
    return Bracketing(l, moved_spans, start_index)


def add(B, x):
    """Helper for binary_bracketings. Adds x to the indices of the brackets.
    """
    return [[(a_b2[0]+x,a_b2[1]+x) for a_b2 in s] for s in B]


def _binary_bracketings(n):
    """Helper for binary_bracketings.
    """
    if n == 1:
        return [[]]
    elif n == 2:
        return [[(0,2)]]
    else:
        b = {}
        for i in range(1, n):
            b[i] = _binary_bracketings(i)
        B = []
        for i in range(1, n):
            # todas las combinaciones posibles de b[i] y add(b[n-i], i):
            b1 = b[i]
            b2 = add(b[n-i], i)
            for j in range(len(b1)):
                for k in range(len(b2)):
                    B = B + [[(0,n)] + b1[j] + b2[k]]
        
        return B


def binary_bracketings(n):
    """Returns all the possible binary bracketings of n leaves.
    """
    # remove whole span bracket and wrap into a Bracketing object:
    return [Bracketing(n, set(b[1:])) for b in _binary_bracketings(n)]


def binary_bracketings_count(n):
    """Returns the number of binary bracketings of n leaves (this is, the
    Catalan number C_{n-1}).
    """
    return catalan(n-1)


def catalan(n):
    """Helper for binary_bracketings_count(n).
    """
    if n <= 1:
        return 1
    else:
        # http://mathworld.wolfram.com/CatalanNumber.html
        return catalan(n-1)*2*(2*n-1)/(n+1)


def rbranch_bracketing(length, start_index=0):
    """Returns the rbranch bracketing of the given length.
    """
    b = set((i, start_index+length) for i in range(start_index+1, start_index+length-1))
    return Bracketing(length, b, start_index=start_index)


def lbranch_bracketing(length, start_index=0):
    """Returns the lbranch bracketing of the given length.
    """
    b = set((start_index, i) for i in range(start_index+2, start_index+length))
    return Bracketing(length, b, start_index=start_index)


def P_split(n):
    """Returns a binary bracketing according to the P_split() distribution.
    n is the number of leaves.
    """
    if n <= 2:
        return Bracketing(n)
    k = random.randint(1, n-1)
    # b = [(0, n)] + gP_split(0, k) + gP_split(k, n)
    b = Bracketing(n, set(gP_split(0, k) + gP_split(k, n)))
    return b


def gP_split(i, j):
    """Helper for P_split().
    """
    if i+1 == j:
        b = []
    else:
        k = random.randint(i+1, j-1)
        b = [(i, j)] + gP_split(i, k) + gP_split(k, j)
    return b


# FIXME: I think it only works with b.start_index = 0.
def P_split_prob(b):
    """Returns the probability of b according to the P_split() distribution.
    """
    """n = b.length
    if n <= 2:
        p = 1.0
    else:
        k = 1
        # si el arbol es binario y n > 2 seguro que tiene que ser splittable.
        #while k < n and not b.splittable(k):
        while not b.splittable(k):
            k += 1
        
        p = (1.0 / float(n)) * gP_split_prob(b, 0, k) * gP_split_prob(b, k, n)
    
    return p"""
    return gP_split_prob(b, b.start_index, b.start_index+b.length)


def gP_split_prob(b, i, j):
    n = j - i
    if n <= 2:
        p = 1.0
    else:
        k = i+1
        # si el arbol es binario y n > 2 seguro que tiene que ser splittable.
        #while k < n and not b.splittable(k):
        while not b.splittable(k, i, j):
            k += 1
        
        p = (1.0 / float(n-1)) * gP_split_prob(b, i, k) * gP_split_prob(b, k, j)
    
    return p
