# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

#!/usr/bin/python

# Calculo de precision y recall para el topline UBOUND


"""
WSJ10
Cantidad de arboles: 7422.0
Medidas sumando todos los brackets:
  Precision: 78.8
  Recall: 100.0
  Media harmonica F1: 88.1
NEGRA10
Cantidad de arboles: 7537.0
Medidas sumando todos los brackets:
  Precision: 56.4
  Recall: 100.0
  Media harmonica F1: 72.1
CAST3LB10
Cantidad de arboles: 712.0
Medidas sumando todos los brackets:
  Precision: 70.1
  Recall: 100.0
  Media harmonica F1: 82.4
"""

from . import model, bracketing

class UBound(model.BracketingModel):
    trained = True
    tested = True
    
    def __init__(self, treebank):
        self.Gold = [bracketing.tree_to_bracketing(t) for t in treebank.trees]
    
    # FIXME: no esta bien adaptado para usar count_fullspan_bracket
    def measures(self, i):
        g = self.Gold[i]
        n = len(g.brackets)
        # m es la cant. de brackets del supuesto parse
        m = g.length - 2
        if m > 0:
            if self.count_fullspan_bracket:
                prec = float(n+1) / float(m+1)
            else:
                prec = float(n) / float(m)
        else:
            prec = 1.0
        rec = 1.0
        return (prec, rec)
    
    def measures2(self, i):
        g = self.Gold[i]
        n = len(g.brackets)
        m = g.length - 2
        if self.count_fullspan_bracket:
            return (n+1, m+1, n+1)
        else:
            return (n, m, n)

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
    m = UBound(tb)
    m.eval()
    return m

def main2():
    from . import negra10
    tb = negra10.Negra10()
    tb.simplify_tags()
    m = UBound(tb)
    m.eval()
    return m

def main3():
    from . import cast3lb10
    tb = cast3lb10.Cast3LB10()
    tb.simplify_tags()
    m = UBound(tb)
    m.eval()
    return m

# VIEJO:

"""wsj10 = wsj.get_wsj10_treebank()

# Recall es 1, obvio.
p = 0.0
r = 1.0
brackets_ok = 0
brackets_parse = 0
brackets_gold = 0
# Cantidad de arboles:
m = 0
for t in wsj10:
    n = len(t.leaves())
    if n >= 3:
        m = m+1
        # print str(m)+"-esima frase..."
        s = t.spannings(leaves=False,root=False,unary=False)
        precision = float(len(s)) / float(n-2)
        brackets_parse += n-2
        brackets_gold += len(s)
        
        p = p + precision
p = p / float(m)
print "Cantidad de arboles:", m
print "Medidas promediando p y r por frase:"
print "  Precision de UBOUND:", p
print "  Recall de UBOUND:", r
print "  Media harmonica F1:", 2*(p*r)/(p+r)
p = float(brackets_gold) / float(brackets_parse)
print "Medidas sumando todos los brackets:"
print "  Precision de UBOUND:", p
print "  Recall de UBOUND:", r
print "  Media harmonica F1:", 2*(p*r)/(p+r)"""

# Cantidad de arboles: 7056
# Medidas promediando p y r por frase:
#   Precision de UBOUND: 0.740901529262
#   Recall de UBOUND: 1.0
#   Media harmonica F1: 0.851169944777
# Medidas sumando todos los brackets:
#   Precision de UBOUND: 0.747252747253
#   Recall de UBOUND: 1.0
#   Media harmonica F1: 0.85534591195

# Intento de usar eval del que desisti antes de fracasar:
# (deberia programar un binarize y que el parse sea eso)
"""import eval

wsj10 = wsj.get_wsj10_treebank()
Gold = []
Parse = []
for t in wsj10:
    if len(t.leaves()) >= 3:
	g = t.spannings(leaves=False,root=False)"""

# Debugging:
"""
Para ir tirando arboles hasta encontrar el que da precision > 1 (UBOUND):
from wsj import *
l = []
m = 0
for e in get_treebank_iterator():
    e.filter_tags()
    n = len(e.leaves())
    if n <= 10:
        m = m+1
        print str(m)+"-esima frase..."
        l = l + [t]
        # Cuento los spans que coinciden no trivialmente con rbranch:
        s = e.spannings(leaves=False)
        s.remove((0,n))
        if len(s) > float(n-2):
            break
"""
