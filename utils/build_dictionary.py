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

def get_vocab(filename):
    v = {}
    fin = codecs.open(filename, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        for word in words:
            w = word.strip()
            if w not in v:
                v[w] = 1
            else:
                v[w] += 1
    
    fin.close()
    return v

def vocab_with_frequency(vocab, f):
    v = {}
    for key in vocab.keys():
        if vocab[key] >= f:
            v[key] = vocab[key]
    return v

def vocab_with_size(vocab, size):
    v_mfw = []
    count = 1
    v = sorted(vocab, key = vocab.get, reverse = True)
    
    for w in v:
        if count <= size:
            v_mfw.append(w.strip())
            count += 1
    return v_mfw

def save_dict(word_freqs, filename, flag=None):
  	words = word_freqs.keys()
        freqs = word_freqs.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()

	if flag is None:
        	worddict['eos'] = 0
        	worddict['UNK'] = 1
        	for ii, ww in enumerate(sorted_words):
            		worddict[ww] = ii+2
	else:
        	for ii, ww in enumerate(sorted_words):
            		worddict[ww] = ii

        with open('%s.json'%filename, 'wb') as f:
            json.dump(worddict, f, indent=2)
            #json.dump(worddict, f, indent=2, ensure_ascii=False)

def _main(file_name, flag, freq=5):
	print "Preparing vocab..."
	vocab = get_vocab(file_name)
	print "Saving vocab:", file_name+'.json'
	save_dict(vocab, file_name, flag)
	print 'SIZE:', len(vocab)
	print "Getting vocab with freq >= 5"
	vocab_wf = vocab_with_frequency(vocab, freq)
	print "Saving vocab:", file_name+'.freq5.json'
	save_dict(vocab_wf, file_name + '.freq5', flag)
	print 'SIZE:', len(vocab_wf)

if __name__ == '__main__':

	if len(sys.argv) != 3:
		print 'Usage: python ', sys.argv[0], 'corpus_file'
		exit()
	_main(sys.argv[1], sys.argv[2])		
