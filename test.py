#!/usr/bin/env python

import os
import sys
import codecs
import gzip
import cPickle
import time
import argparse
import subprocess
import numpy 
import json
import glob

from os.path import basename
from collections import OrderedDict
from rnn import GRU_test
from rnn import LSTM_test
from utils.tools import minibatch, contextwin, shuffle, add_padding
from utils.test_preprocess import preprocess_data
from metrics.qe_eval import wmt_eval
from metrics.pos_eval import icon_eval

select_model = {
		"GRU_adadelta_bilingual_pretrain": GRU_test.GRU_bilingual,
		"GRU_adadelta_bilingual": GRU_test.GRU_bilingual, 
		"GRU_adadelta_pretrain": GRU_test.GRU,
		"GRU_adadelta": GRU_test.GRU,
		"GRU_pretrain": GRU_test.GRU,
		"GRU": GRU_test.GRU,
		"GRU_adadelta_char_bilingual_pretrain": GRU_test.GRU_char_bilingual,
		"GRU_adadelta_char_bilingual": GRU_test.GRU_char_bilingual, 
		"GRU_adadelta_char_pretrain": GRU_test.GRU_char,
		"GRU_adadelta_char": GRU_test.GRU_char,
		"LSTM_adadelta_bilingual_pretrain": LSTM_test.LSTM_bilingual,
		"LSTM_adadelta_bilingual": LSTM_test.LSTM_bilingual, 
		"LSTM_adadelta_pretrain": LSTM_test.LSTM,
		"LSTM_adadelta": LSTM_test.LSTM,
		"LSTM_pretrain": LSTM_test.LSTM,
		"LSTM": LSTM_test.LSTM,
		}

def merg_dicts(x, y):
	z = x.copy()
	z.update(y)
	return z

def _write_text(text, filename):
	with codecs.open(filename, 'w', 'utf-8') as fout:
		for line in text:
			fout.write(' '.join(line) + '\n')

def load(model_dir):
	flist = glob.glob(model_dir + '/*.npy')
	names = []
	for fname in flist:
		fname = basename(fname)
		names.append(os.path.splitext(fname)[0])
    	params = OrderedDict((p, q) for p, q in zip(names, names))
    
	'''
	Read the saved model
	'''
    	for name, value in params.items():
        	pp = numpy.load(os.path.join(model_dir, name + '.npy'))
        	params[name] = pp
	return params

def test(data_test=['data/qe/test/test.src.lc',
                'data/qe/test/test.mt.lc',
                'data/qe/test/test.align'],
          data_test_y = 'data/qe/test/test.tags',
          dictionaries=['data/qe/train/train.src.lc.json',
              'data/qe/train/train.mt.lc.json'],
          character2index=['data/qe/train/train.src.lc.dict_char.json',
              'data/qe/train/train.mt.lc.dict_char.json'],
          embeddings=['data/qe/pretrain/ep_qe.en.vector.txt',
              'data/qe/pretrain/ep_qe.de.vector.txt'],
	  label2index = 'data/qe/train/train.tags.json',
	  load_model = None
    ):
	model_dir = load_model[0]
	current_options = OrderedDict(sorted(locals().copy().items()))
	train_options = json.load(open(model_dir+'/model.json'))
	model_options = merg_dicts(current_options, train_options)

	print 'model_options:', model_options

	model_name = model_options['use_model'][0]
	if model_options['use_adadelta']:
		model_name += '_adadelta'
	if model_options['use_char']:
		model_name += '_char'
	if model_options['use_bilingual']:
		model_name += '_bilingual'
	if model_options['use_pretrain']:
		model_name += '_pretrain'

	print 'Using model:', model_name

	processed_data = preprocess_data(
		data_test=model_options['data_test'], data_test_y=model_options['data_test_y'][0], 
		dictionaries=model_options['dictionaries'],
		embeddings = model_options['embeddings'],
		character2index = model_options['character2index'],
		label2index = model_options['label2index'][0],
		use_bilingual=model_options['use_bilingual'], 
		use_char=model_options['use_char'], 
		use_pretrain=model_options['use_pretrain'])

	test, test_y, w2idxs, char2idxs, label2idxs, embs = processed_data
	idx2label = dict((k,v) for v,k in label2idxs.iteritems())

        vocsize_s = vocsize_t = vocsize_schar = vocsize_tchar = 0
        emb_s, emb_t, test_s, test_schar, test_t, test_tchar, = ([] for i in range(6))

        if (model_options['use_bilingual'] or len(test) == 4) and model_options['use_char']:
                emb_s, emb_t = embs
                test_s, test_t, test_schar, test_tchar = test
                vocsize_s = len(w2idxs[0])
                vocsize_t = len(w2idxs[1])
                vocsize_schar = len(char2idxs[0])
                vocsize_tchar = len(char2idxs[1])

        elif model_options['use_char']:
                emb_t = embs[0]
                test_t, test_tchar = test
                vocsize_t = len(w2idxs[0])
                vocsize_tchar = len(char2idxs[0])

	elif model_options['use_bilingual'] or len(test) == 2:
                emb_s, emb_t = embs
                test_s, test_t = test
                vocsize_s = len(w2idxs[0])
                vocsize_t = len(w2idxs[1])
        else :
                emb_t = embs[0]
                test_t = test[0]
                vocsize_t = len(w2idxs[0])

    	numpy.random.seed(model_options['seed'])
    	# instanciate the model

	params = load(model_dir)

    	rnn = select_model[model_name](
                    de = model_options['dim_word'],
                    cs = model_options['win'],
                    de_char = model_options['dim_char'],
                    max_char = model_options['max_char'],
		    params = params)

	predictions_test, groundtruth_test, predictions_valid,groundtruth_valid = ([] for i in range(4))

   	if model_options['use_bilingual'] and model_options['use_char']:
                        predictions_test = [ map(lambda x: idx2label[x],
                                rnn.classify(numpy.asarray(contextwin(x,
                                model_options['win'])).astype('int32'),
                                numpy.asarray(contextwin(_x, model_options['win'])).astype('int32'),
                                numpy.asarray(add_padding(__x,
                                model_options['max_char'])).astype('int32')))
                                for x, _x, __x in zip(test_s, test_t, test_tchar) ]

	elif model_options['use_bilingual']:
		#evaluation // back into the real world : idx -> words
            	predictions_test = [ map(lambda x: idx2label[x],
                	rnn.classify(numpy.asarray(contextwin(x_src, 
			model_options['win'])).astype('int32'),
			numpy.asarray(contextwin(x_tgt,model_options['win'])).astype('int32')))
			for x_src, x_tgt in zip(test_s, test_t) ]

	elif model_options['use_char']:
                        predictions_test = [ map(lambda x: idx2label[x],
                                rnn.classify(numpy.asarray(contextwin(x,
                                model_options['win'])).astype('int32'),
                                numpy.asarray(add_padding(_x,
                                model_options['max_char'])).astype('int32')))
                                for x, _x, in zip(test_t, test_tchar) ]
	else:
                #evaluation // back into the real world : idx -> words
                predictions_test = [ map(lambda x: idx2label[x],
			rnn.classify(numpy.asarray(contextwin(x,
			model_options['win'])).astype('int32'))) for x in test_t ]

	if data_test_y:
                groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
                #words_test = [ map(lambda x: idx2word[x], w) for w in test_t]

        #evaluation // compute the accuracy using conlleval.pl
	res_test = []
	output_file = model_dir + '/test_output.txt'
	if model_options['use_quest']:
		if data_test_y:
		   print '\nWriting the output into:', output_file
		   res_test=wmt_eval(predictions_test, groundtruth_test, output_file)
		else:
		   print '\nWriting the output into:', output_file
		   _write_text(predictions_test, output_file)
	if model_options['use_tag']:
		if data_test_y:
		   print '\nWriting the output into:', output_file
                   res_test=icon_eval(predictions_test, groundtruth_test, output_file)
		else:
		   print '\nWriting the output into:', output_file
		   _write_text(predictions_test, output_file)

	if data_test_y:
        	print 'test F1' , res_test , ' '*20

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	data = parser.add_argument_group('data sets; model loading and saving')
	data.add_argument('--data_test', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel test corpus (source, target and alignment)")
	data.add_argument('--data_test_y', type=str, required=False, metavar='PATH', nargs=1,
                         help="test labels")
	data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (source and target vocabulary)")
	data.add_argument('--character2index', type=str, required=True, metavar='PATH', nargs="+",
                         help="character vocabularies (source and target vocabulary)")
	data.add_argument('--label2index', type=str, required=True, metavar='PATH', nargs="+",
                         help="network target labels")
	data.add_argument('--load_model', type=str, required=True, metavar='PATH', nargs=1,
                         help="path to the trained model ")
	data.add_argument('--embeddings', type=str, metavar='PATH', nargs="+",
                         help="network vocabularies (source and target vocabulary)")
	args = parser.parse_args()

	#print vars(args)
	test(**vars(args))
