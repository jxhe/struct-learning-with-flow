# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

from nltk.parse import dependencygraph
from nltk import tree

import treebank

class DepGraph(dependencygraph.DependencyGraph):

    def __init__(self, nltk_depgraph):
        dependencygraph.DependencyGraph.__init__(self)
        self.nodelist = nltk_depgraph.nodelist
        self.root = nltk_depgraph.root
        self.stream = nltk_depgraph.stream

    def remove_leaves(self, f):
        """f must be a function that takes a node dict and returns a boolean.
        """
        nodelist = self.nodelist
        newnodelist = [nodelist[0].copy()]
        newindex = [0]
        i, j = 1, 1
        while i < len(nodelist):
            node = nodelist[i]
            if not f(node):
                # this node stays
                newnode = node.copy()
                newnode['address'] = j
                newnodelist.append(newnode)
                newindex.append(j)
                j += 1
            else:
                newindex.append(-1)
            i += 1
        #print newindex
        # fix attributes 'head' and 'deps':
        node = newnodelist[0]
        node['deps'] = [newindex[i] for i in node['deps'] if newindex[i] != -1]
        for node in newnodelist[1:]:
            i = newindex[node['head']]
            if i == -1:
                raise Exception('Removing non-leaf.')
            node['head'] = i
            node['deps'] = [newindex[i] for i in node['deps'] if newindex[i] != -1]
        self.nodelist = newnodelist
    
    def constree(self):
        # Some depgraphs have several roots (for instance, 512th of Turkish).
        #i = self.root['address']
        roots = self.nodelist[0]['deps']
        if len(roots) == 1:
            return treebank.Tree(self._constree(roots[0]))
        else:
            # TODO: check projectivity here also.
            trees = [self._constree(i) for i in roots]
            return treebank.Tree(tree.Tree('TOP', trees))
    
    def _constree(self, i):
        node = self.nodelist[i]
        word = node['word']
        deps = node['deps']
        if len(deps) == 0:
            t = tree.Tree(node['tag'], [word])
            t.span = (i, i+1)
            return t
        address = node['address']
        ldeps = [j for j in deps if j < address]
        rdeps = [j for j in deps if j > address]
        lsubtrees = [self._constree(j) for j in ldeps]
        rsubtrees = [self._constree(j) for j in rdeps]
        csubtree = tree.Tree(node['tag'], [word])
        csubtree.span = (i, i+1)
        subtrees = lsubtrees+[csubtree]+rsubtrees
        
        # check projectivity:
        for j in range(len(subtrees)-1):
            if subtrees[j].span[1] != subtrees[j+1].span[0]:
                raise Exception('Non-projectable dependency graph.')
        
        t = tree.Tree(word, subtrees)
        j = subtrees[0].span[0]
        k = subtrees[-1].span[1]
        t.span = (j, k)
        return t


def from_depset(depset, s):
    """Returns a DepGraph with the dependencies of depset over the sentence s.
    (i, j) in depset means that s[i] depends on s[j]. depset must be sorted.
    """
    tab = ""
    for i, j in depset:
        tab += '\t'.join([str(i+1), s[i], '_', s[i], '_\t_', str(j+1), '_\t_\t_\n'])
    return DepGraph(dependencygraph.DependencyGraph(tab))
