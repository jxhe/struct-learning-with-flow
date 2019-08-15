# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# sentence.py: Class Sentence.

import string

class Sentence: # (list):
    
    def __init__(self, tag_list):
        if isinstance(tag_list, str):
            tag_list = tag_list.split()
        self.tag_list = tag_list
        # si heredara de list podria poner:
        #list.__init__(self, tag_list)
    
    def __str__(self):
        return string.join(self.tag_list)
    
    def __repr__(self):
        return str(self)
    
    def reverse(self):
        self.tag_list.reverse()
    
    # iterador sobre todas las subsecuencias de tags.
    # devuelve las subsecuencias separadas por espacios en un string.
    def itersubseqs(self):
        l = len(self.tag_list)
        x = 2 # span minimo
        for i in range(x, l+1):
            for j in range(l-i+1):
                yield string.join(self.tag_list[j:j+i])
    
    # iterador sobre todos los contextos "a la CCM".
    # devuelve pares de tags.
    def itercontexts(self):
        s = self.tag_list + ['END', 'START']
        l = len(self.tag_list)
        x = 2 # span minimo
        for i in range(x, l+1):
            for j in range(l-i+1):
                yield (s[j-1], s[j+i])
    
    # Por francolq, basdo en tree.Tree de NLTK 0.9:
    
    #////////////////////////////////////////////////////////////
    # Disabled list operations
    #////////////////////////////////////////////////////////////
    
    def __rmul__(self, v):
        raise TypeError('Sentence does not support multiplication')
   
    #////////////////////////////////////////////////////////////
    # Enabled list operations
    #///////////////////////////////////////////////////////////
    
    # ver "Emulating numeric types":
    # http://docs.python.org/ref/numeric-types.html
    def __add__(self, v):
        return Sentence(self.tag_list + v)
    def __radd__(self, v):
        return Sentence(v + self.tag_list)
    def __iadd__(self, v):
        self.tag_list += v
        return self
    def __mul__(self, v):
        return Sentence(self.tag_list * v)
   
    #////////////////////////////////////////////////////////////
    # Indexing
    #////////////////////////////////////////////////////////////
    
    def __len__(self):
        return len(self.tag_list)
   
    def __getitem__(self, index):
        # return list.__getitem__(self, index)
        return self.tag_list[index]
   
    def __setitem__(self, index, value):
        self.tag_list[index] = value
   
    # se me hace que no la necesito:
    def __delitem__(self, index):
        del(self.tag_list[index])
