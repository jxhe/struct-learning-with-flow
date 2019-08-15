# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# lbranch.py: LBRANCH baseline for unsupervised parsing.

from . import bracketing
from . import model

class LBranch(model.BracketingModel):
    trained = True
    tested = True
    
    def __init__(self, treebank=None):
        model.BracketingModel.__init__(self, treebank)
        self.Parse = [bracketing.lbranch_bracketing(b.length) for b in self.Gold]


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
    m = LBranch(tb)
    m.eval()

def main2():
    from . import negra10
    tb = negra10.Negra10()
    tb.simplify_tags()
    m = LBranch(tb)
    m.eval()

def main3():
    from . import cast3lb10
    tb = cast3lb10.Cast3LB10()
    tb.simplify_tags()
    m = LBranch(tb)
    m.eval()

"""
from lbranch import *
main()

WSJ10
Cantidad de arboles: 7422.0
Medidas sumando todos los brackets:
  Precision: 25.7
  Recall: 32.6
  Media harmonica F1: 28.7
NEGRA10
Cantidad de arboles: 7537.0
Medidas sumando todos los brackets:
  Precision: 27.4
  Recall: 48.6
  Media harmonica F1: 35.1
CAST3LB10
Cantidad de arboles: 712.0
Medidas sumando todos los brackets:
  Precision: 26.9
  Recall: 38.4
  Media harmonica F1: 31.7
"""
