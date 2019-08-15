# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# -*- coding: iso-8859-1 -*-
# Creado en base al modulo nltk_lite.corpora.treebank.

# Natural Language Toolkit: Penn Treebank Reader
#
# Copyright (C) 2001-2005 University of Pennsylvania
# Author: Steven Bird <sb@ldc.upenn.edu>
#         Edward Loper <edloper@gradient.cis.upenn.edu>
# URL: <http://nltk.sf.net>
# For license information, see LICENSE.TXT

# from nltk_lite.corpora import get_basedir

import itertools

from nltk import tree

from . import treebank

# Funciona para el Cast3LB antes y despues de quitar las hojas, 
# y antes y despues de eliminar las funciones.
def is_ellipsis(s):
    # 'sn.co' aparece como tag de una elipsis en '204_c-3.tbf', 11, 
    # 'sn' en 'a12-4.tbf', 1 y en 'a14-0.tbf', 2.
    return s == '*0*' or \
            s.split('-')[0] in ['sn.e', 'sn.e.1', 'sn.co', 'sn']


# Funciona para el Cast3LB solo si las hojas son POS tags.
def is_punctuation(s):
    return s.lower()[0] == 'f'


class Cast3LBTree(treebank.Tree):
    
    
    # Funciona para el Cast3LB antes y despues de quitar las hojas, 
    # y antes y despues de eliminar las funciones.
    def is_ellipsis(self, s):
        return is_ellipsis(s)
    
    
    # Funciona para el Cast3LB solo si las hojas son POS tags.
    def is_punctuation(self, s):
        return is_punctuation(s)


class Cast3LB(treebank.SavedTreebank):
    default_basedir = "3lb-cast"
    trees = []
    filename = 'cast3lb.treebank'
    
    
    def __init__(self, basedir=None, load=False):
        if basedir == None:
            self.basedir = self.default_basedir
        else:
            self.basedir = basedir
        if load:
            self.get_trees()
    
    
    # Devuelve el arbol que se encuentra en la posicion offset de los archivos 
    # files del treebank Cast3LB. Sin parametros devuelve un arbol cualquiera.
    # files puede ser un nombre de archivo o una lista de nombres de archivo.
    def get_tree(self, files=None, offset=0):
        # Parsear files y parar cuando se llegue al item offset+1.
        #t = [t for t in itertools.islice(parsed(files),offset+1)][offset]
        #if preprocess:
        #    t = prepare(t)
        #return t
        t = self.get_trees2(files, offset, offset+1)[0]
        return t
    
    
    # Devuelve los arboles que se encuentran en la posicion i con start <= i < end
    # dentro de los archivo files del treebank Cast3LB.
    # files puede ser un nombre de archivo o una lista de nombres de archivo.
    def get_trees2(self, files=None, start=0, end=None):
        lt = [t for t in itertools.islice(self.parsed(files), start, end)]
        return lt
    
    
    """# puede ser reemplazado en las subclases para filtrar:
    # FIXME: capaz que get_trees2 hace lo mismo y esto es al pedo:
    def _generate_trees(self):
        print "Parseando el Cast3LB treebank..."
        trees = [self._prepare(t) for t in self.parsed()]
        return trees
    
    
    # para ser reemplazado en las subclases:
    def _prepare(self, t):
        return t"""
    
    
    def remove_ellipsis(self):
        list(map(lambda t: t.remove_ellipsis(), self.trees))
    
    
    def remove_punctuation(self):
        list(map(lambda t: t.remove_punctuation(), self.trees))
    
    
    def parsed(self, files=None):
        for t in treebank.SavedTreebank.parsed(self, files):
            yield Cast3LBTree(tree.Tree('ROOT', [t]), t.labels)
    
    
    # Funciona para el Cast3LB antes y despues de quitar las hojas, 
    # y antes y despues de eliminar las funciones.
    def is_ellipsis(self, s):
        return is_ellipsis(s)
    
    
    # Funciona para el Cast3LB solo si las hojas son POS tags.
    def is_punctuation(self, s):
        return is_punctuation(s)


"""# Devuelve el treebank Cast3LB entero.
def get_treebank():
    cast3lb_treebank = treebank.load_treebank('cast3lb.treebank')
    if cast3lb_treebank is None:
    return cast3lb_treebank

# Devuelve los datos de entrenamiento del Cast3LB.
def get_training_treebank():
    training_treebank = treebank.load_treebank('cast3lb_training.treebank')
    if training_treebank is None:
        print "Parseando datos de entrenamiento del Cast3LB treebank..."
        training_files = get_training_files()
        trees = [prepare(t) for t in parsed(training_files)]
        training_treebank = treebank.Treebank(trees)
        training_treebank.save('cast3lb_training.treebank')
    return training_treebank

# Devuelve los datos de testeo del Cast3LB.
def get_test_treebank():
    test_treebank = treebank.load_treebank('cast3lb_test.treebank')
    if test_treebank is None:
        print "Parseando datos de testeo del Cast3LB treebank..."
        test_files = get_test_files()
        trees = [prepare(t) for t in parsed(test_files)]
        test_treebank = treebank.Treebank(trees)
        # Ordena de menor a mayor largo de oracion.
        test_treebank.length_sort()
        test_treebank.save('cast3lb_test.treebank')
    return test_treebank
"""

"""def get_files(filename):
    f = open(filename, 'r')
    s = f.read()
    return s.split()


def get_training_files(filename='training.txt'):
    return get_files(filename)


def get_test_files(filename='test.txt'):
    return get_files(filename)
"""

"""
def prepare(t):
    ""
    Prepara un arbol obtenido del Cast3LB para ser usado para crear un 
    PCFG.
    
    @param t: el arbol
    ""
    t.remove_leaves()
    return Cast3LBTree(tree.Tree('ROOT', [t]), t.labels)
"""


"""def filter_nodes(t, f):
    if not isinstance(t, tree.Tree):
        return t

    subtrees = []
    for st in t:
        if (isinstance(st, tree.Tree) and f(st.node)) or \
           (not isinstance(st, tree.Tree) and f(st)):
            st = filter_nodes(st, f)
            subtrees += [st]
    return tree.Tree(t.node, subtrees)"""

"""
Raw:

    Pierre Vinken, 61 years old, will join the board as a nonexecutive
    director Nov. 29.

Tagged:

    Pierre/NNP Vinken/NNP ,/, 61/CD years/NNS old/JJ ,/, will/MD join/VB 
    the/DT board/NN as/IN a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD ./.

NP-Chunked:

    [ Pierre/NNP Vinken/NNP ]
    ,/, 
    [ 61/CD years/NNS ]
    old/JJ ,/, will/MD join/VB 
    [ the/DT board/NN ]
    as/IN 
    [ a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD ]
    ./.

Parsed:

    ( (S 
      (NP-SBJ 
        (NP (NNP Pierre) (NNP Vinken) )
        (, ,) 
        (ADJP 
          (NP (CD 61) (NNS years) )
          (JJ old) )
        (, ,) )
      (VP (MD will) 
        (VP (VB join) 
          (NP (DT the) (NN board) )
          (PP-CLR (IN as) 
            (NP (DT a) (JJ nonexecutive) (NN director) ))
          (NP-TMP (NNP Nov.) (CD 29) )))
      (. .) ))
"""


"""def chunked(files = 'chunked'):
    ""
    @param files: One or more treebank files to be processed
    @type files: L{string} or L{tuple(string)}
    @rtype: iterator over L{tree}
    ""       

    # Just one file to process?  If so convert to a tuple so we can iterate
    if isinstance(files, str):
        files = (files,)

    for file in files:
        path = os.path.join(get_basedir(), "treebank", file)
        s = open(path).read()
        for t in tokenize.blankline(s):
            yield tree.chunk(t)


def tagged(files = 'chunked'):
    ""
    @param files: One or more treebank files to be processed
    @type files: L{string} or L{tuple(string)}
    @rtype: iterator over L{list(tuple)}
    ""       

    # Just one file to process?  If so convert to a tuple so we can iterate
    if isinstance(files, str):
        files = (files,)

    for file in files:
        path = os.path.join(get_basedir(), "treebank", file)
        f = open(path).read()
        for sent in tokenize.blankline(f):
            l = []
            for t in tokenize.whitespace(sent):
                if (t != '[' and t != ']'):
                    l.append(tag2tuple(t))
            yield l

def raw(files = 'raw'):
    ""
    @param files: One or more treebank files to be processed
    @type files: L{string} or L{tuple(string)}
    @rtype: iterator over L{list(string)}
    ""       

    # Just one file to process?  If so convert to a tuple so we can iterate
    if isinstance(files, str):
        files = (files,)

    for file in files:
        path = os.path.join(get_basedir(), "treebank", file)
        f = open(path).read()
        for sent in tokenize.blankline(f):
            l = []
            for t in tokenize.whitespace(sent):
                l.append(t)
            yield l


def demo():
    from nltk_lite.corpora import treebank

    print "Parsed:"
    for tree in itertools.islice(treebank.parsed(), 3):
        print tree.pp()
    print

    print "Chunked:"
    for tree in itertools.islice(treebank.chunked(), 3):
        print tree.pp()
    print

    print "Tagged:"
    for sent in itertools.islice(treebank.tagged(), 3):
        print sent
    print

    print "Raw:"
    for sent in itertools.islice(treebank.raw(), 3):
        print sent
    print

if __name__ == '__main__':
    demo()
"""