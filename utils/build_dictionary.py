#!/usr/bin/env python

"""
Createb by Raj Nath Patel, April 7 2017
Purpose: To prepare dictionary for RNN vocabulary

Usage: python  scripts/prep_dict.py corpus_file save_dict_to
Eg: python scripts/prep_dict.py neuralsum/cnn/clean/text/complete.txt cnn_dict
"""
import json, numpy

from collections import OrderedDict
import sys, re, codecs, os, gzip, cPickle

def get_vocab(filename):
    v = {}
    fin = codecs.open(filename, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        for word in words:
            w = word.strip().lower()
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

        with open('%s.json'%filename, 'wb') as f:
            json.dump(worddict, f, indent=2)
            #json.dump(worddict, f, indent=2, ensure_ascii=False)

def _main(file_name, freq=5):
	print "Preparing vocab..."
	vocab = get_vocab(file_name)
	print "Saving vocab..."
	save_dict(vocab, file_name)
	print len(vocab)
	print "Getting vocab with freq >= 5"
	vocab_wf = vocab_with_frequency(vocab, freq)
	print "Saving vocab..."
	save_dict(vocab_wf, file_name + '.freq5')
	print len(vocab_wf)

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print 'Usage: python ', sys.argv[0], 'corpus_file'
		exit()
	_main(sys.argv[1])		
