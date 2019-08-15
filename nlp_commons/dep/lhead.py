# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# lhead.py: LHEAD baseline for unsupervised dependency parsing.

from dep import model
from dep import depset

class LHead(model.DepModel):
    trained = True
    tested = True
    
    def __init__(self, treebank=None):
        model.DepModel.__init__(self, treebank)
        self.Parse = [depset.lhead_depset(b.length) for b in self.Gold]


def main():
    print "WSJ10"
    import dep.dwsj
    tb = dep.dwsj.DepWSJ10()
    m = LHead(tb)
    m.eval()

"""
from dep import lhead
lhead.main()

WSJ10
Number of Trees: 7422
  Directed Accuracy: 23.7
  Undirected Accuracy: 55.6
Debe dar: 24.0, 55.9.

>>> m.count_length_2_1 = True
>>> m.eval()
Number of Trees: 7422
  Directed Accuracy: 24.0
  Undirected Accuracy: 55.7
(52248, 0.23974123411422446, 0.55726534986985143)
Mas cerca...
"""
