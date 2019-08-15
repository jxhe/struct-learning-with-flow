# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# rbranch.py: RBRANCH baseline for unsupervised parsing.

from . import bracketing, model

class RBranch(model.BracketingModel):
    trained = True
    tested = True
    
    def __init__(self, treebank=None):
        model.BracketingModel.__init__(self, treebank)
        self.Parse = [bracketing.rbranch_bracketing(b.length) for b in self.Gold]


def main():
    print('WSJ10')
    main1()
    print('NEGRA10')
    main2()
    print('CAST3LB10')
    main3()

def main1():
    from . import wsj10
    tb = wsj10.WSJ10()
    m = RBranch(tb)
    m.eval()

def main2():
    from . import negra10
    tb = negra10.Negra10()
    tb.simplify_tags()
    m = RBranch(tb)
    m.eval()

def main3():
    from . import cast3lb10
    tb = cast3lb10.Cast3LB10()
    tb.simplify_tags()
    m = RBranch(tb)
    m.eval()

"""
from rbranch import *
main()

WSJ10
Cantidad de arboles: 7422.0
Medidas sumando todos los brackets:
  Precision: 55.2
  Recall: 70.0
  Media harmonica F1: 61.7
NEGRA10
Cantidad de arboles: 7537.0
Medidas sumando todos los brackets:
  Precision: 33.9
  Recall: 60.1
  Media harmonica F1: 43.3
CAST3LB10
Cantidad de arboles: 712.0
Medidas sumando todos los brackets:
  Precision: 46.9
  Recall: 67.0
  Media harmonica F1: 55.2
"""

# VIEJO:

# No hace falta construir los parses binarios RBRANCH.
"""p = 0.0
r = 0.0
brackets_ok = 0
brackets_parse = 0
brackets_gold = 0
# Cantidad de arboles:
m = 0
for b in bs:
    n = b.length
    #if n >= 3:
    if True:
        m = m+1
        # print str(m)+"-esima frase..."
        # s = t.spannings(leaves=False,root=False,unary=False)
        # s2 = filter(lambda (a,b): b == n, s)
        s = b.brackets
        s2 = filter(lambda (a,b): b == n+1, s)
        
        precision = float(len(s2)) / float(n-2)
        
        if len(s) > 0:
            recall = float(len(s2)) / float(len(s))
        else:
            recall = 1.0
        
        brackets_ok += len(s2)
        brackets_parse += n-2
        brackets_gold += len(s)
        
        p = p + precision
        r = r + recall
p = p / float(m)
r = r / float(m)
print "Cantidad de arboles:", m
print "Medidas promediando p y r por frase:"
print "  Precision de RBRANCH:", p
print "  Recall de RBRANCH:", r
print "  Media harmonica F1:", 2*(p*r)/(p+r)
p = float(brackets_ok) / float(brackets_parse)
r = float(brackets_ok) / float(brackets_gold)
print "Medidas sumando todos los brackets:"
print "  Precision de RBRANCH:", p
print "  Recall de RBRANCH:", r
print "  Media harmonica F1:", 2*(p*r)/(p+r)"""

# Debugging:
"""
Para ir tirando arboles hasta encontrar el que da recall con division por 0 (RBRANCH):
from wsj import *
l = []
m = 0
for e in get_treebank_iterator():
    e.filter_tags()
    n = len(e.leaves())
    if n <= 10:
        m = m+1
        print str(m)+"-esima frase..."
        l = l + [e]
        if m == 100:
            break
"""
