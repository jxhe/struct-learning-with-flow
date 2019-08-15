# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

import itertools

from . import cast3lb

class Cast3LBn(cast3lb.Cast3LB):
    
    
    def __init__(self, n, basedir=None, load=True):
        cast3lb.Cast3LB.__init__(self, basedir)
        self.n = n
        self.filename = 'cast3lb%02i.treebank' % n
        if load:
            self.get_trees()
    
    
    def _generate_trees(self):
        print("Parsing Cast3LB treebank...")
        f = lambda t: len(t.leaves()) <= self.n
        m = lambda t: self._prepare(t)
        trees = [t for t in filter(f, map(m, self.parsed()))]
        return trees
    
    
    def _prepare(self, t):
        t.remove_leaves()
        t.remove_ellipsis()
        t.remove_punctuation()
        return t
    
    
    def simplify_tags(self):
        list(map(lambda t: t.map_leaves(self.tag_filter), self.trees))
    
    
    def tag_filter(self, t):
        t2 = t.lower()
        if t == '':
            print("Empty tag!", t)
            return t
        elif t2[0] == 'a':
            # Adjetivo: Dejo tipo (calificativo 'q' u ordinal 'o') y
            # numero (singular 's', plural 'p' o invariable 'n').
            return t2[0:2]+t2[4]
        elif t2[0] == 'r':
            # Adverbio: solo hay 'rg' y 'rn' para la palabra 'no'. Lo dejo asi.
            return t2
        elif t2[0] == 'd':
            # Determinante: Dejo tipo y numero.
            return t2[0:2]+t2[4]
        elif t2[0] == 'n':
            # Nombre: Dejo tipo y numero.
            return t2[0:2]+t2[3]
        elif t2[0] == 'v':
            # Verbo: dejo tipo, modo y numero.
            return t2[0:3]+t2[5]
        elif t2 in ['i', 'y', 'zm', 'zp', 'z', 'w', 'cc', 'cs', 'x']:
            # [interjeccion, abreviatura, moneda, porcentaje, numero, fecha u hora,
            # conjuncion coordinada, conjuncion subordinada, elemento deconocido].
            return t2
        elif t2[0:2] == 'sp':
            # Adposicion de tipo preposicion: dejo forma.
            return t2[0:3]
        elif t2[0] == 'p':
            # Pronombre: dejo tipo y numero.
            return t2[0:2]+t2[4]
        elif t2[0] == 'f':
            # Puntuacion. La devolvemos sin pasar a lowercase.
            return t
        else:
            print("Unrecognized tag:", t)
            return t
        # Quedan colgados los tags: sn, sn.e.1, sn.co
    
    
    def simplify_tags_more(self):
        list(map(lambda t: t.map_leaves(self.tag_filter_more), self.trees))
    
    
    def tag_filter_more(self, t):
        t2 = t.lower()
        if t == '':
            print("Empty tag!", t)
            return t
        elif t2[0] == 'a':
            # Adjetivo: Dejo numero (singular 's', plural 'p' o invariable 'n').
            return t2[0]+t2[4]
        elif t2[0] == 'r':
            # Adverbio: solo hay 'rg' y 'rn' para la palabra 'no'. Lo dejo asi.
            return t2
            # Unifico todos:
            #return t[0]
        elif t2[0] == 'd':
            # Determinante: Dejo numero.
            return t2[0]+t2[4]
        elif t2[0] == 'n':
            # Nombre: Dejo tipo (comun o propio) y numero.
            return t2[0:2]+t2[3]
        elif t2[0] == 'v':
            # Verbo: dejo modo y numero.
            return t2[0]+t2[2]+t2[5]
        elif t2 in ['i', 'y', 'zm', 'zp', 'z', 'w', 'cc', 'cs', 'x']:
            # [interjeccion, abreviatura, moneda, porcentaje, numero, fecha u hora,
            # conjuncion coordinada, conjuncion subordinada, elemento deconocido].
            return t2
        elif t2[0:2] == 'sp':
            # Adposicion de tipo preposicion: no dejo nada.
            return t2[0:2]
        elif t2[0] == 'p':
            # Pronombre: dejo numero.
            return t2[0]+t2[4]
        elif t2[0] == 'f':
            # Puntuacion. La devolvemos sin pasar a lowercase.
            return t
        else:
            print("Unrecognized tag:", t)
            return t
        # Quedan colgados los tags: sn, sn.e.1, sn.co


class Cast3LB10(Cast3LBn):
    
    
    def __init__(self, basedir=None, load=True):
        Cast3LBn.__init__(self, 10, basedir, load)


class Cast3LB30(Cast3LBn):
    
    
    def __init__(self, basedir=None, load=True):
        Cast3LBn.__init__(self, 30, basedir, load)


class Cast3LBPn(Cast3LBn):
    # sadly I need this list (redundant beacuse we have is_punctuation) to allow 
    # usage by other classes that want to pickle this information:
    punctuation_tags = ['Fp', 'Fs', 'Fpa', 'Fia', 'Fit', 'Fx', 'Fz', 'Fat', 'Fpt', 'Fc', 'Fd', 'Fe', 'Fg', 'Faa']
    # this was found this way:
    #from cast3lb10 import *
    #tb = Cast3LB10P()
    #punct = set(sum(([x for x in t.leaves() if tb.is_punctuation(x)] for t in tb.trees), []))
    
    stop_punctuation_tags = ['Fp', 'Fs', 'Fx', 'Fz', 'Fc', 'Fd', 'Fe', 'Fg']
    bracket_punctuation_tag_pairs = [('Fpa', 'Fpt'), ('Fia', 'Fit'), ('Faa', 'Fat')]
    # these are: parenthesis, question marks, exclamation marks
    # quotes appear all with the Fe tag.
    # other not present in Cast3LB10P: Fc*: [ ], Fr*: << >>, Fl*: { }
    
    
    def __init__(self, n, basedir=None, load=True):
        #Cast3LB10n.__init__(self, n, load=False)
        if basedir == None:
            self.basedir = self.default_basedir
        else:
            self.basedir = basedir
        
        self.n = n
        self.filename = 'cast3lb%02ip.treebank' % n
        if load:
            self.get_trees()
    
    
    def _generate_trees(self):
        print("Parsing Cast3LB treebank...")
        f = lambda t: len([x for x in t.leaves() if not cast3lb.is_punctuation(x)]) <= self.n
        m = lambda t: self._prepare(t)
        trees = [t for t in filter(f, map(m, self.parsed()))]
        return trees
    
    
    def _prepare(self, t):
        t.remove_leaves()
        t.remove_ellipsis()
        #t.remove_punctuation()
        return t


# XXX: For consistency This class should be called Cast3LBP10:
class Cast3LB10P(Cast3LBPn):
    
    
    def __init__(self, basedir=None, load=True):
        Cast3LBPn.__init__(self, 10, basedir, load)


class Cast3LBP30(Cast3LBPn):
    
    
    def __init__(self, basedir=None, load=True):
        Cast3LBPn.__init__(self, 30, basedir, load)


def test():
    tb = Cast3LB10()
    tb.simplify_tags()
    return tb
