# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# paramdict.py: ParamDict is a comfortable dictionary with commonly needed
# functions.

class ParamDict(object):

    def __init__(self, d=None, default_val=0.0, count_evidence=False):
        if d is None:
            self.d = {}
        else:
            self.d = d
        self.count_evidence = count_evidence
        if count_evidence:
            self.evidence = {}
        self.default_val = default_val

    def set_default_val(self, val):
        self.default_val = val

    def val(self, x):
        return self.d.get(x, self.default_val)

    def setVal(self, x, val):
        self.d[x] = val

    def add1(self, x):
        self.add(x, 1.0)

    def add(self, x, y):
        add(self.d, x, y)
        if self.count_evidence and y > 0.0:
            add(self.evidence, x, 1.0)

    def iteritems(self):
        return iter(self.d.items())


# Common procedure used in ParamDict:
def add(dict, x, val):
    dict[x] = dict.get(x, 0) + val
