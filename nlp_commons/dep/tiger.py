
import re

import nltk
from nltk.corpus.reader import api

import treebank

basedir = 'corpora/TiGerDB'

class Tiger10(api.SyntaxCorpusReader):

    def __init__(self):
        api.SyntaxCorpusReader.__init__(self, nltk.data.find(basedir), 'fdsc-Apr08/.*\.fdsc')

    def _read_block(self, stream):
        return [stream.readlines()]
        #return [stream.read()]
        #s = stream.readline()
        #while not s.startswith('sentence_form'):
        #    s = stream.readline()
    
    def _word(self, s):
        # jump to sentence:
        i = 0
        while i < len(s) and not s[i].startswith('sentence_form('):
            i += 1
        assert i < len(s)
        l = s[i]
        
        return l[14:-3].split()

    def _tag(self, s, simplify_tags=False):
        return [(x, x) for x in self._word(s)]

    def _parse(self, s):
        #print s

        # get sentence length:
        w = self._word(s)
        n = len(w)

        # jump to structure:
        i = 0
        while i < len(s) and not s[i].startswith('structure('):
            i += 1
        assert i < len(s)

        # read dependencies:
        deps = []
        i += 1
        while i < len(s) and not s[i].startswith(')'):
            l = s[i]
            #print 'Empieza con', l
            l2 = [x for x in re.split(r'[\(,~\s\)]*', l) if x != '']
            if len(l2) == 5:
                # this line encodes a dependency
                j = int(l2[4])
                k = int(l2[2])
                if j <= n and k <= n:
                    deps += [(j, k)]
            else:
                assert len(l2) == 4
            i += 1
        assert i < len(s)

        deps.sort()

        return deps
        
