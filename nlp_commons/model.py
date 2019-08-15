# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# model.py: A general model for parsing (class Model).
# Also a general model for bracketing parsing (class BracketingModel).

import itertools
import sys

from . import util
from . import sentence
from . import bracketing
from . import wsj10

class Model:
    Gold = []
    Parse = []
    evaluation = None
    trained = False
    tested = False
    evaluated = False
    
    def train(self):
        self.trained = True
    
    def parse(self, s):
        return None
    
    #def test(self, S):
    #    self.Parse = [self.parse(s) for s in S]
    #    self.tested = True
    
    def test(self, short=False, max_length=None):
        self.Parse, self.Weight = [], 0.0

        #n = str(len(self.S))
        #m = len(n)
        #o = "%"+str(m)+"d of "+n
        #i = 0
        #print "Parsed", o % i,
        #sys.stdout.flush()
        #o = ("\b"*(2*m+5)) + o
        p = util.Progress('Parsed', 0, len(self.S))
        for s in self.S:
            if max_length is None or len(s) <= max_length:
                (parse, weight) = self.parse(s)
            else:
                (parse, weight) = (None, 0.0)
            self.Parse += [parse]
            self.Weight += weight
            #i += 1
            #print o % i,
            #sys.stdout.flush()
            next(p)
        print("\nFinished parsing.")
        self.eval(short=short, max_length=max_length)
        self.tested = True
    
    def eval(self, short=False, max_length=None):
        self.evaluated = True


class BracketingModel(Model):
    count_fullspan_bracket = True
    count_length_2 = True
    count_length_2_1 = False
    
    def __init__(self, treebank=None, training_corpus=None):
        
        treebank = self._get_treebank(treebank)
        if training_corpus == None:
            training_corpus = treebank
        self.training_corpus = training_corpus
        
        S, Gold = [], []
        #for s in treebank.sents():
        for s in treebank.tagged_sents():
            s = [x[1] for x in s]
            S += [sentence.Sentence(s)]
        
        for t in treebank.parsed_sents():
            Gold += [bracketing.tree_to_bracketing(t)]
        
        self.S = S
        self.Gold = Gold
    
    def _get_treebank(self, treebank=None):
        if treebank is None:
            treebank = wsj10.WSJ10()
        return treebank
    
    def eval(self, output=True, short=False, long=False, max_length=None):
        """Compute precision, recall and F1 between the parsed bracketings and
        the gold bracketings.
        """
        Gold = self.Gold
        
        Prec = 0.0
        Rec = 0.0
        
        # Medidas sumando brackets y despues promediando:
        brackets_ok = 0
        brackets_parse = 0
        brackets_gold = 0
        
        for i in range(len(Gold)):
            l = Gold[i].length
            if (max_length is None or l <= max_length) \
                    and (self.count_length_2_1 or (self.count_length_2 and l == 2) or l >= 3):
                (prec, rec) = self.measures(i)
                Prec += prec
                Rec += rec
                
                # Medidas sumando brackets y despues promediando:
                (b_ok, b_p, b_g) = self.measures2(i)
                brackets_ok += b_ok
                brackets_parse += b_p
                brackets_gold += b_g
        
        m = float(len(Gold))
        Prec2 = float(brackets_ok) / float(brackets_parse)
        Rec2 = float(brackets_ok) / float(brackets_gold)
        F12 = 2*(Prec2*Rec2)/(Prec2+Rec2)
        
        self.evaluation = (m, Prec2, Rec2, F12)
        self.evaluated = True
        
        if output and not short:
            #print "Cantidad de arboles:", int(m)
            #print "Medidas sumando todos los brackets:"
            #print "  Precision: %2.1f" % (100*Prec2)
            #print "  Recall: %2.1f" % (100*Rec2)
            #print "  Media harmonica F1: %2.1f" % (100*F12)
            #if long:
                #print "Brackets parse:", brackets_parse
                #print "Brackets gold:", brackets_gold
                #print "Brackets ok:", brackets_ok
                #Prec = Prec / m
                #Rec = Rec / m
                #F1 = 2*(Prec*Rec)/(Prec+Rec)
                #print "Medidas promediando p y r por frase:"
                #print "  Precision: %2.1f" % (100*Prec)
                #print "  Recall: %2.1f" % (100*Rec)
                #print "  Media harmonica F1: %2.1f" % (100*F1)
            print("Sentences:", int(m))
            print("Micro-averaged measures:")
            print("  Precision: %2.1f" % (100*Prec2))
            print("  Recall: %2.1f" % (100*Rec2))
            print("  Harmonic mean F1: %2.1f" % (100*F12))
            if int:
                print("Brackets parse:", brackets_parse)
                print("Brackets gold:", brackets_gold)
                print("Brackets ok:", brackets_ok)
                Prec = Prec / m
                Rec = Rec / m
                F1 = 2*(Prec*Rec)/(Prec+Rec)
                print("Macro-averaged measures:")
                print("  Precision: %2.1f" % (100*Prec))
                print("  Recall: %2.1f" % (100*Rec))
                print("  Harmonic mean F1: %2.1f" % (100*F1))
        elif output and short:
            print("F1 =", F12)
        
        return self.evaluation
    
    # FIXME: no esta bien adaptado para usar count_fullspan_bracket
    # Funcion auxiliar de eval();
    # Precision y recall del i-esimo parse respecto de su gold:
    def measures(self, i):
        g = self.Gold[i].brackets
        if self.Parse[i] is None:
            p, n = set(), 0
        else:
            p = self.Parse[i].brackets
            n = float(bracketing.coincidences(self.Gold[i], self.Parse[i]))
        
        if len(p) > 0:
            if self.count_fullspan_bracket:
                prec = (n+1) / float(len(p)+1)
            else:
                prec = n / float(len(p))
        elif len(g) == 0:
            prec = 1.0
        else:
            # XXX: no deberia ser 1?
            prec = 0.0
        
        if len(g) > 0:
            if self.count_fullspan_bracket:
                rec = (n+1) / float(len(g)+1)
            else:
                rec = n / float(len(g))
        else:
            rec = 1.0
        
        return (prec, rec)
    
    # FIXME: hacer andar con frases de largo 1!
    # devuelve la terna (brackets_ok, brackets_parse, brackets_gold)
    # del i-esimo arbol. Se usa para calcular las medidas 
    # micro-promediadas.
    def measures2(self, i):
        g = self.Gold[i].brackets
        if self.Parse[i] is None:
            p, n = set(), 0
        else:
            p = self.Parse[i].brackets
            n = float(bracketing.coincidences(self.Gold[i], self.Parse[i]))
        if self.count_fullspan_bracket:
            return (n+1, len(p)+1, len(g)+1)
        else:
            return (n, len(p), len(g))
   
    # FIXME: pegado asi nomas: adaptar esto para usar measures.
    def eval_by_length(self):
        #Prec = {}
        #Rec = {}
        Gold = self.Gold
        Parse = self.Parse
        
        brackets_ok = {}
        brackets_parse = {}
        brackets_gold = {}
        
        for i in range(2, 11):
            brackets_ok[i] = 0
            brackets_parse[i] = 0
            brackets_gold[i] = 0
        
        for gb, pb in zip(Gold, Parse):
            gb.set_start_index(0)
            pb.set_start_index(0)
            l = gb.length
            for i in range(2, l):
                g = set([x_y for x_y in gb.brackets if x_y[1]-x_y[0] == i])
                p = set([x_y1 for x_y1 in pb.brackets if x_y1[1]-x_y1[0] == i])
                
                brackets_ok[i] += len(g & p)
                brackets_parse[i] += len(p)
                brackets_gold[i] += len(g)
            if self.count_fullspan_bracket and ((self.count_length_2 and l == 2) or l >= 3):
                brackets_ok[l] += 1
                brackets_parse[l] += 1
                brackets_gold[l] += 1
            
        Prec = {}
        Rec = {}
        F1 = {}
        print("i\tP\tR\tF1")
        for i in range(2, 10):
            Prec[i] = float(brackets_ok[i]) / float(brackets_parse[i])
            Rec[i] = float(brackets_ok[i]) / float(brackets_gold[i])
            F1[i] = 2*(Prec[i]*Rec[i])/(Prec[i]+Rec[i])
            print("%i\t%2.2f\t%2.2f\t%2.2f" % (i, 100*Prec[i], 100*Rec[i], 100*F1[i]))
    
        return (Prec, Rec, F1)
