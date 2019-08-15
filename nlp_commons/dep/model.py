# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt


from __future__ import absolute_import
# dep/model.py: A general model for dependency parsing (class DepModel), and a
# general model for projective dependency parsing, with evaluation also as
# constituent trees.


#from .. import model
from .. import sentence, bracketing, model
from . import depset
from . import dwsj


class DepModel(model.Model):
    """A general model for dependency parsing."""
    count_length_2 = True
    count_length_2_1 = False

    def __init__(self, treebank=None):

        treebank = self._get_treebank(treebank)

        S, Gold = [], []
        for t in treebank.get_trees():
            s = sentence.Sentence(t.leaves())
            S += [s]
            #Gold += [depset.deptree_to_depset(t)]
            Gold += [t.depset]

        self.S = S
        self.Gold = Gold

    def _get_treebank(self, treebank=None):
        if treebank is None:
            treebank = dwsj.DepWSJ10()
        return treebank

    def eval(self, output=True, short=False, long=False, max_length=None):
        Gold = self.Gold

        Count = 0
        Directed = 0.0
        Undirected = 0.0

        for i in range(len(Gold)):
            l = Gold[i].length
            if (max_length is None or l <= max_length) \
                    and (self.count_length_2_1 or (self.count_length_2 and l == 2) or l >= 3):
                (count, directed, undirected) = self.measures(i)
                Count += count
                Directed += directed
                Undirected += undirected

        Directed = Directed / Count
        Undirected = Undirected / Count

        self.evaluation = (Count, Directed, Undirected)
        self.evaluated = True

        if output and not short:
            print "Number of Trees:", len(Gold)
            print "  Directed Accuracy: %2.1f" % (100*Directed)
            print "  Undirected Accuracy: %2.1f" % (100*Undirected)
        elif output and short:
            print "L =", Directed, "UL =", Undirected

        return self.evaluation

    def measures(self, i):
        # Helper for eval().
        # Measures for the i-th parse.

        g, p = self.Gold[i].deps, self.Parse[i].deps
        (n, d, u) = (self.Gold[i].length, 0, 0)
        for (a, b) in g:
            b1 = (a, b) in p
            b2 = (b, a) in p
            if b1:
                d += 1
            if b1 or b2:
                u += 1

        return (n, d, u)

    #def eval_stats(self, output=True, short=False, long=False, max_length=None):
    def eval_stats(self, output=True, max_length=None):
        Gold, Parse = self.Gold, self.Parse
        gold_stats = {}
        parse_stats = {}
        stats = {}
        for i in range(len(Gold)):
            l = Gold[i].length
            if (max_length is None or l <= max_length) \
                    and (self.count_length_2_1 or (self.count_length_2 and l == 2) or l >= 3):
                #(count, directed, undirected) = self.measures(i)
                #Count += count
                #Directed += directed
                #Undirected += undirected
                s = self.S[i] + ['ROOT']
                g, p = Gold[i].deps, Parse[i].deps
                lg = [(s[i], s[j], i < j) for i,j in g]
                lp = [(s[i], s[j], i < j) for i,j in p]
                for x in lg:
                    gold_stats[x] = gold_stats.get(x, 0) + 1
                    stats[x] = stats.get(x, 0) - 1
                for x in lp:
                    parse_stats[x] = parse_stats.get(x, 0) + 1
                    stats[x] = stats.get(x, 0) + 1
        lstats = sorted(stats.iteritems(), key=lambda x:x[1])
        if output:
            # a -> b iif b is head of a.
            print 'Overproposals'
            for ((d, h, left), n) in lstats[:len(lstats)-10:-1]:
                if left:
                    print '\t{0} -> {1}\t{2}'.format(d, h, n)
                else:
                    print '\t{1} <- {0}\t{2}'.format(d, h, n)
            print 'Underproposals'
            for ((d, h, left), n) in lstats[:10]:
                if left:
                    print '\t{0} -> {1}\t{2}'.format(d, h, -n)
                else:
                    print '\t{1} <- {0}\t{2}'.format(d, h, -n)

        #return (gold_stats, parse_stats)
        return lstats


class ProjDepModel(DepModel):
    """A general model for projective dependency parsing, with evaluation also
    as constituent trees.
    """
    def __init__(self, treebank=None, training_corpus=None):
        """
        The elements of the treebank must be trees with a DepSet in the
        attribute depset.
        """
        treebank = self._get_treebank(treebank)
        if training_corpus == None:
            training_corpus = treebank
        self.test_corpus = treebank
        self.training_corpus = training_corpus
        S = []
        for s in treebank.tagged_sents():
            s = [x[1] for x in s]
            S += [sentence.Sentence(s)]
        self.S = S
        # Extract gold as DepSets:
        # FIXME: call super and do this there.
        self.Gold = [t.depset for t in treebank.parsed_sents()]

        # Extract gold as Bracketings:
        # self.bracketing_model = model.BracketingModel(treebank)

    def eval(self, output=True, short=False, long=False, max_length=None):
        """Compute evaluation of the parses against the test corpus. Computes
        unlabeled precision, recall and F1 between the bracketings, and directed
        and undirected dependency accuracy between the dependency structures.
        """
        # XXX: empezamos a lo bruto:
        self.bracketing_model.Parse = [bracketing.tree_to_bracketing(t) for t in self.Parse]
        #dmvccm.DMVCCM.eval(self, output, short, long, max_length)
        self.bracketing_model.eval(output, short, long, max_length)

        # Ahora eval de dependencias:
        self.DepParse = self.Parse
        # type no anda porque devuelve instance:
        #self.Parse = [type(self).tree_to_depset(t) for t in self.DepParse]
        self.Parse = [self.__class__.tree_to_depset(t) for t in self.DepParse]
        #model.DepModel.eval(self, output, short, long, max_length)
        DepModel.eval(self, output, short, long, max_length)
        self.Parse = self.DepParse

    def eval_stats(self, output=True, max_length=None):
        # Ahora eval de dependencias:
        self.DepParse = self.Parse
        # type no anda porque devuelve instance:
        #self.Parse = [type(self).tree_to_depset(t) for t in self.DepParse]
        self.Parse = [self.__class__.tree_to_depset(t) for t in self.DepParse]
        #model.DepModel.eval(self, output, short, long, max_length)
        DepModel.eval_stats(self, output, max_length)
        self.Parse = self.DepParse

    @staticmethod
    def tree_to_depset(t):
        """Function used to convert the trees returned by the parser to DepSets.
        """
        raise Exception('Static function tree_to_depset must be overriden.')
