#!/usr/bin/env python

import sys
import re
import codecs
import os
import gzip
import cPickle
import json
import numpy

from collections import OrderedDict

def get_char_vocab(filename):
    v = {}
    fin = codecs.open(filename, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        for word in words:
            w = word.strip()
	    for ch in list(w):
            	if ch not in v:
                	v[ch] = 1
            	else:
                	v[ch] += 1
    
    fin.close()
    return v

def save_dict(word_freqs, filename):
  	words = word_freqs.keys()
        freqs = word_freqs.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()

        worddict['eos'] = 0
        worddict['UNK'] = 1
        for ii, ww in enumerate(sorted_words):
            	worddict[ww] = ii+2

        with open('%s.dict_char.json'%filename, 'wb') as f:
            json.dump(worddict, f, indent=2)
            #json.dump(worddict, f, indent=2, ensure_ascii=False)

def _main(file_name):
	print "Preparing vocab..."
	vocab = get_char_vocab(file_name)
	print "Saving vocab:", file_name+'.json'
	save_dict(vocab, file_name)
	print 'SIZE:', len(vocab)

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print 'Usage: python ', sys.argv[0], 'corpus_file'
		exit()
	_main(sys.argv[1])		
