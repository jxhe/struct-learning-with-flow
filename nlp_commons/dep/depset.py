# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# depset.py: Dependency set.

from .. import util


class DepSet:

    def __init__(self, length, deps):
        self.length = length
        self.deps = deps


def from_depgraph(g):
    length = len(g.nodelist)-1
    deps = [(n['address']-1, n['head']-1) for n in g.nodelist[1:]]
    return DepSet(length, deps)


def from_string(s):
    """
    >>> d = from_string('[(0,3), (1,0), (2,1), (3,-1)]\n')
    """
    """t = s[1:].split()
    l = len(t)
    deps = []
    for x in t:
        y = x[1:-2].split(',')
        deps += [(int(y[0]), int(y[1]))]"""
    deps = util.safe_eval(s)
    l = len(deps)
    return DepSet(l, deps)


def deptree_to_depset(t):
    return DepSet(len(t.leaves()), t.depset)


def lhead_depset(length):
    deps = [(i, i-1) for i in range(length)]
    return DepSet(length, deps)


def rhead_depset(length):
    deps = [(i, i+1) for i in range(length-1)] + [(length-1, -1)]
    return DepSet(length, deps)


def _binary_depsets(n):
    """Helper for binary_depsets.
    """
    if n == 0:
        return [[]]
    elif n == 1:
        return [[(0, -1)]]
    else:
        result = []
        for i in range(n):
            lres = _binary_depsets(i)
            rres = _binary_depsets(n-1-i)
            lres = map(lambda l: [(j, (k!=-1 and k) or i) for (j,k) in l], lres)
            rres = map(lambda l: [(j+i+1, (k!=-1 and (k+i+1)) or i) for (j,k) in l], rres)
            #print i, lres, rres
            result += [l+[(i, -1)]+r for l in lres for r in rres]

        return result


def binary_depsets(n):
    """Returns all the binary dependency trees for a sentence of length n.
    """
    return map(lambda s: DepSet(n, s), _binary_depsets(n))


def _all_depsets(n):
    """Helper for all_depsets.
    """
    # Dynamically programmed:
    depsets = {1: [[(0, -1)]]}
    sums = _all_sums(n)
    sums[0] = [[]]

    for i in range(2, n+1):
        result = []
        for j in range(0, i):
            # j is the root.

            # to the left:
            lres = []
            ll = sums[j]
            for l in ll:
                laux = [[]]
                acum = 0
                for k in l:
                    # for instance, j=3, l=[1,2].
                    laux2 = []
                    for m in depsets[k]:
                        m2 = [(p+acum, (q!=-1 and (q+acum)) or j) for (p,q) in m]
                        laux2 += [m2]
                    laux = [o+m2 for o in laux for m2 in laux2]
                    acum += k
                lres += laux

            # to the right:
            rres = []
            ll = sums[i-1-j]
            for l in ll:
                laux = [[]]
                acum = j+1
                for k in l:
                    laux2 = []
                    for m in depsets[k]:
                        m2 = [(p+acum, (q!=-1 and (q+acum)) or j) for (p,q) in m]
                        laux2 += [m2]
                    laux = [o+m2 for o in laux for m2 in laux2]
                    acum += k
                rres += laux

            #lres = map(lambda l: [(p, (q!=-1 and q) or j) for (p,q) in l], lres)
            #rres = map(lambda l: [(p+j+1, (q!=-1 and (q+j+1)) or j) for (p,q) in l], rres)

            result += [l+[(j, -1)]+r for l in lres for r in rres]
        depsets[i] = result

    return depsets


    """if n == 0:
        return [[]]
    elif n == 1:
        return [[(0, -1)]]
    else:
        result = []
        for i in range(n):
            lres = [[]]
            for lsplits in range(0, i):

                lres = _all_depsets(i)


            rres = _all_depsets(n-1-i)
            lres = map(lambda l: [(j, (k!=-1 and k) or i) for (j,k) in l], lres)
            rres = map(lambda l: [(j+i+1, (k!=-1 and (k+i+1)) or i) for (j,k) in l], rres)
            #print i, lres, rres
            result += [l+[(i, -1)]+r for l in lres for r in rres]

        return result"""


def all_depsets(n):
    """Returns all the dependency sets for a sentence of length n.
    """
    return map(lambda s: DepSet(n, s), _all_depsets(n))


def _all_sums(n):
    """Helper for all_depsets.
    Returns a dictionary with keys 1, ..., n.
    The value for each i is a list with all the ways of summing i.
    """
    # Dynamically programmed
    # sums(1) = 1
    # sums(2) = 1+1,2
    # sums(3) = 1+1+1,1+2,2+1,3
    # sums(4) = 1+1+1+1,1+1+2,1+2+1,2+1+1,2+2,1+3,3+1,4
    #         = map (1+) (sums(3)), map (2+) (sums(2)), map (3+) (sums(1)), 4
    #         = 1+1+1+1,1+1+2,1+2+1,1+3,  2+1+1,2+2,  3+1,   4
    sums = {1:[[1]]}
    for i in range(2, n+1):
        l = []
        for j in range(1, i):
            l += [[j]+l2 for l2 in sums[i-j]]
        l += [[i]]
        sums[i] = l

    return sums
