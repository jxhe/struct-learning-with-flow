# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt



from . import bracketing

count_fullspan_bracket = True
count_length_2 = True
count_length_2_1 = False

# Calculo de precision, recall y F1 para dos Bracketings:
def eval(Gold, Parse, output=True, short=False, long=False):
    assert len(Gold) == len(Parse)

    # Medidas sumando brackets y despues promediando:
    brackets_ok = 0
    brackets_parse = 0
    brackets_gold = 0
    
    for gb, pb in zip(Gold, Parse):
        l = gb.length
        if count_length_2_1 or (count_length_2 and l == 2) or l >= 3:
            # Medidas sumando brackets y despues promediando:
            (b_ok, b_p, b_g) = measures(gb, pb)
            brackets_ok += b_ok
            brackets_parse += b_p
            brackets_gold += b_g
            
            """# Medidas sumando brackets y despues promediando:
            brackets_ok += n
            brackets_parse += len(p)
            brackets_gold += len(g)"""
    
    m = float(len(Gold))
    Prec = float(brackets_ok) / float(brackets_parse)
    Rec = float(brackets_ok) / float(brackets_gold)
    F1 = 2*(Prec*Rec)/(Prec+Rec)
    if output and not short:
        print("Cantidad de arboles:", m)
        print("Medidas sumando todos los brackets:")
        print("  Precision: %2.1f" % (100*Prec))
        print("  Recall: %2.1f" % (100*Rec))
        print("  Media harmonica F1: %2.1f" % (100*F1))
        if int:
            print("Brackets parse:", brackets_parse)
            print("Brackets gold:", brackets_gold)
            print("Brackets ok:", brackets_ok)
    elif output and short:
        print("F1 =", F1)
    else:
        return (m, Prec, Rec, F1)


def string_measures(gs, ps):
    gb = bracketing.string_to_bracketing(gs)
    pb = bracketing.string_to_bracketing(ps)
    return measures(gb, pb)


# FIXME: hacer andar con frases de largo 1!
# devuelve la terna (brackets_ok, brackets_parse, brackets_gold)
# del i-esimo arbol. Se usa para calcular las medidas 
# micro-promediadas.
def measures(gb, pb):
    g, p = gb.brackets, pb.brackets
    n = bracketing.coincidences(gb, pb)
    if count_fullspan_bracket:
        return (n+1, len(p)+1, len(g)+1)
    else:
        return (n, len(p), len(g))


# TODO: esta funcion es util, podria pasar a model.BracketingModel.
# goldtb debe ser un treebank, parse una lista de bracketings.
def eval_label(label, goldtb, parse):
    Rec = 0.0
    brackets_ok = 0
    brackets_gold = 0
    
    bad = []
    
    for gt, pb in zip(goldtb.trees, parse):
        g = set(x[1] for x in gt.labelled_spannings(leaves=False, root=False, unary=False) if x[0] == label)
        gb = bracketing.Bracketing(pb.length, g, start_index=0)
        
        n = bracketing.coincidences(gb, pb)
        if len(g) > 0:
            rec = float(n) / float(len(g))
            bad += [difference(gb, pb)]
        else:
            rec = 1.0
            bad += [set()]
        Rec += rec
        
        brackets_ok += n
        brackets_gold += len(g)
        
    m = len(parse)
    Rec = Rec / float(m)
    
    print("Recall:", Rec)
    print("Brackets gold:", brackets_gold)
    print("Brackets ok:", brackets_ok)
    
    return (Rec, bad)

# Conj. de brackets que estan en b1 pero no en b2
# los devuelve con indices comenzando del 0.
def difference(b1, b2):
    s1 = set([(x_y[0] - b1.start_index, x_y[1] - b1.start_index) for x_y in b1.brackets])
    s2 = set([(x_y1[0] - b2.start_index, x_y1[1] - b2.start_index) for x_y1 in b2.brackets])
    return s1 - s2


    