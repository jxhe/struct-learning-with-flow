# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# rhead.py: RHEAD baseline for unsupervised dependency parsing.

from dep import model
from dep import depset

class RHead(model.DepModel):
    trained = True
    tested = True
    
    def __init__(self, treebank=None):
        model.DepModel.__init__(self, treebank)
        self.Parse = [depset.rhead_depset(b.length) for b in self.Gold]


def main():
    print "WSJ10"
    import dep.dwsj
    tb = dep.dwsj.DepWSJ10()
    m = RHead(tb)
    m.eval()

"""
from dep import rhead
rhead.main()

WSJ10
Number of Trees: 7422
  Directed Accuracy: 33.5
  Undirected Accuracy: 56.4
"""
