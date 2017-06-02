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

from collections import OrderedDict
from rnn import GRU
from rnn import LSTM
from utils.tools import minibatch, contextwin, shuffle, add_padding
from utils.data_preprocess import preprocess_data
from metrics.qe_eval import wmt_eval
from metrics.pos_eval import icon_eval

select_model = {
		"GRU_adadelta_char_bilingual_pretrain": GRU.GRU_adadelta_char_bilingual_pretrain,
		"GRU_adadelta_char_bilingual": GRU.GRU_adadelta_char_bilingual,
		"GRU_adadelta_char": GRU.GRU_adadelta_char,
		"GRU_adadelta_char_pretrain": GRU.GRU_adadelta_char_pretrain,
		"GRU_adadelta_bilingual_pretrain": GRU.GRU_adadelta_bilingual_pretrain,
		"GRU_adadelta_bilingual": GRU.GRU_adadelta_bilingual, 
		"GRU_adadelta_pretrain": GRU.GRU_adadelta_pretrain,
		"GRU_adadelta": GRU.GRU_adadelta,
		"GRU_pretrain": GRU.GRU_pretrain,
		"GRU": GRU.GRU,
		"LSTM": LSTM.LSTM,
		"LSTM_pretrain": LSTM.LSTM_pretrain,
		"LSTM_adadelta": LSTM.LSTM_adadelta,
		"LSTM_adadelta_pretrain": LSTM.LSTM_adadelta_pretrain,
		"LSTM_adadelta_bilingual": LSTM.LSTM_adadelta_bilingual,
		"LSTM_adadelta_bilingual_pretrain": LSTM.LSTM_adadelta_bilingual_pretrain,
		}

def train(dim_word=100,  # word vector dimensionality
          dim_char=10,  # the number of LSTM units
          max_char=10,  # the number of LSTM units
          dim=100,  # the number of LSTM units
	  win=5, #Window size
	  bs=5, #number of backprop through time steps
	  seed=123,
	  verbose=1,
          use_model='GRU', #Choose the model from- LSTM, DEEPLSTM, RNN, 
          patience=10,  # early stopping patience
          max_epochs=50,
          lrate=0.0005,  # learning rate
          maxlen=100,  # maximum length of the description
          data_train=['data/qe/train/train.src.lc',
              'data/qe/train/train.mt.lc',
              'data/qe/train/train.align'],
          data_train_y = 'data/qe/train/train.tags',
          data_valid=['data/qe/dev/dev.src.lc',
                'data/qe/dev/dev.mt.lc',
                'data/qe/dev/dev.align'],
          data_valid_y = 'data/qe/dev/dev.tags',
          data_test=['data/qe/test/test.src.lc',
                'data/qe/test/test.mt.lc',
                'data/qe/test/test.align'],
          data_test_y = 'data/qe/test/test.tags',
          dictionaries=['data/qe/train/train.src.lc.json',
              'data/qe/train/train.mt.lc.json'],
          character2index=['data/qe/train/train.src.lc.dict_char.json',
              'data/qe/train/train.mt.lc.dict_char.json'],
	  label2index = 'data/qe/train/train.tags.json',
          embeddings=['data/qe/pretrain/ep_qe.en.vector.txt',
              'data/qe/pretrain/ep_qe.de.vector.txt'],
	  use_adadelta=False,
          use_bilingual=False,
          use_pretrain=False,
          use_quest=False,
          use_tag=False,
          use_char=False,
          saveto=False,
          shuffle_each_epoch=True,
	  load_data=None,
    ):

	model_options = OrderedDict(sorted(locals().copy().items()))
	print 'Model_Options:', model_options

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

	processed_data = []
	if load_data:
	    with gzip.open(load_data[0],'rb') as fp:
			processed_data = cPickle.load(fp)
	else:
	    processed_data = preprocess_data(data_train=model_options['data_train'], 
		data_train_y=model_options['data_train_y'][0],
		data_valid=model_options['data_valid'], data_valid_y=model_options['data_valid_y'][0], 
		data_test=model_options['data_test'], data_test_y=model_options['data_test_y'][0], 
		dictionaries=model_options['dictionaries'],
		character2index=model_options['character2index'],
		label2index = model_options['label2index'][0],
		embeddings = model_options['embeddings'],
		use_bilingual=model_options['use_bilingual'], 
		use_char=model_options['use_char'], 
		use_pretrain=model_options['use_pretrain'])

	"""
	Savinn the model/data with model_name
	"""
	save_data = folder = ''
	if use_tag:
		save_data = 'tag.data_' + model_name + '.pkl.gz'
		folder = 'tag.' + model_name
	if use_quest:
		save_data = 'quest.data_' + model_name + '.pkl.gz'
		folder = 'quest.' + model_name

	if saveto:
		with gzip.open(save_data,'wb') as fp:
       			cPickle.dump(processed_data, fp)
    	if not os.path.exists(folder): os.mkdir(folder)

	train, train_y, test, test_y, valid, valid_y, w2idxs, char2idxs, label2idxs, embs=processed_data
	idx2label = dict((k,v) for v,k in label2idxs.iteritems())
	#print len(train), len(test), len(valid)

	vocsize_s = vocsize_t = vocsize_schar = vocsize_tchar = 0
        emb_s, emb_t, train_s, train_schar, train_t, train_tchar, test_s, test_schar, test_t, test_tchar, valid_s, valid_schar, valid_t, valid_tchar = ([] for i in range(14))
		
	if (use_bilingual or len(train) == 4) and use_char:
		emb_s, emb_t = embs
		train_s, train_t, train_schar, train_tchar = train
		test_s, test_t, test_schar, test_tchar = test
		valid_s, valid_t, valid_schar, valid_tchar = valid
    		vocsize_s = len(w2idxs[0])
    		vocsize_t = len(w2idxs[1])
		vocsize_schar = len(char2idxs[0])
		vocsize_tchar = len(char2idxs[1])

	elif use_char:
		emb_t = embs[0]
		train_t, train_tchar = train
		test_t, test_tchar = test
		valid_t, valid_tchar = valid
    		vocsize_t = len(w2idxs[0])
		vocsize_tchar = len(char2idxs[0])

	elif use_bilingual or len(train) == 2:
		emb_s, emb_t = embs
		train_s, train_t = train
		test_s, test_t = test
		valid_s, valid_t = valid
    		vocsize_s = len(w2idxs[0])
    		vocsize_t = len(w2idxs[1])
	else :
		emb_t = embs[0]
		train_t = train[0]
		test_t = test[0]
		valid_t = valid[0]
    		vocsize_t = len(w2idxs[0])

    	nclasses = len(label2idxs)
    	nsentences = len(train_t)

    	numpy.random.seed(model_options['seed'])
    	# instanciate the model
    	rnn = select_model[model_name]( nh = model_options['dim'],
                    nc = nclasses,
                    de = model_options['dim_word'],
                    cs = model_options['win'],
                    de_char = model_options['dim_char'],
		    ne_char = vocsize_tchar,
		    ne_src = vocsize_s,
		    ne_tgt = vocsize_t,
		    emb_src = emb_s,
		    emb_tgt = emb_t,
		    max_char = model_options['max_char'])

    	# train with early stopping on validation set
    	best_f1 = -numpy.inf
    	model_options['patience'] = 2
    	batch_size = (nsentences/100) * 10
    	n_batches = nsentences//batch_size
    	print n_batches
    	for e in xrange(model_options['max_epochs']):
	  model_options['ce'] = e
      	  #shuffle
	  if shuffle_each_epoch:
      	  	shuffle([train_t, train_s, train_tchar, train_y], model_options['seed'])

      	  tic = time.time()
      	  for k in xrange(n_batches):
            #Creating batches
	    batch_train_s = []
	    batch_train_char = []

	    if model_options['use_bilingual']:
            	batch_train_s = train_s[k*batch_size:(k+1)*batch_size]
	    if model_options['use_char']:
            	batch_train_char = train_tchar[k*batch_size:(k+1)*batch_size]

            batch_train_t = train_t[k*batch_size:(k+1)*batch_size]
            batch_train_y = train_y[k*batch_size:(k+1)*batch_size]
            batch_err = 0
            for i in xrange(batch_size):
		cwords_src = []
		padded_chars = []
		if model_options['use_bilingual']:
                	cwords_src = contextwin(batch_train_s[i], model_options['win'])
		if model_options['use_char']:
			padded_chars = add_padding(batch_train_char[i], model_options['max_char'])

		#print batch_train_char[0]
		#print padded_chars
                cwords_tgt = contextwin(batch_train_t[i], model_options['win'])
                labels = batch_train_y[i]

		if model_options['use_bilingual'] and model_options['use_char']:
                     err = rnn.train_grad_shared(cwords_src, cwords_tgt, padded_chars, labels, model_options['lrate'])
		elif model_options['use_char']:
                     err = rnn.train_grad_shared(cwords_tgt, padded_chars, labels, model_options['lrate'])
		elif model_options['use_bilingual']:
                     err = rnn.train_grad_shared(cwords_src, cwords_tgt, labels, model_options['lrate'])
		elif model_options['use_adadelta']:
                     err = rnn.train_grad_shared(cwords_tgt, labels, model_options['lrate'])
		else:
		     err = rnn.train(cwords_tgt, labels, model_options['lrate'])
                
		if model_options['use_adadelta']:
		     rnn.train_update(model_options['lrate'])

                rnn.normalize()
                
                if model_options['verbose']:
                    print '[learning] epoch %i batch %i >> %2.2f%%'%(e, k, (i+1)*100./batch_size),'completed in %.2f (sec) <<\r'%(time.time()-tic),
		    sys.stdout.flush()

	    if(k % model_options['patience'] == 0):

		predictions_test, groundtruth_test, predictions_valid, \
			groundtruth_valid = ([] for i in range(4))

		if model_options['use_bilingual'] and model_options['use_char']:
			predictions_test = [ map(lambda x: idx2label[x],
				rnn.classify(numpy.asarray(contextwin(x, 
				model_options['win'])).astype('int32'),
				numpy.asarray(contextwin(_x, model_options['win'])).astype('int32'),
				numpy.asarray(add_padding(__x, 
				model_options['max_char'])).astype('int32')))
				for x, _x, __x in zip(test_s, test_t, test_tchar) ]
                	groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
                	#words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

                	predictions_valid = [ map(lambda x: idx2label[x],
                                 rnn.classify(numpy.asarray(contextwin(x, 
				 model_options['win'])).astype('int32'),
				 numpy.asarray(contextwin(_x, model_options['win'])).astype('int32'),
				 numpy.asarray(add_padding(__x, 
				 model_options['max_char'])).astype('int32')))
                                 for x, _x, __x in zip(valid_s, valid_t, valid_tchar) ]
                	groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]

		elif model_options['use_bilingual']:
			#evaluation // back into the real world : idx -> words
            		predictions_test = [ map(lambda x: idx2label[x],
                                 rnn.classify(numpy.asarray(contextwin(x_src, 
				 model_options['win'])).astype('int32'),
                                 numpy.asarray(contextwin(x_tgt,model_options['win'])).astype('int32')))
                                 for x_src, x_tgt in zip(test_s, test_t) ]
            		groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
           		#words_test = [ map(lambda x: idx2word_de[x], w) for w in test_lex]

            		predictions_valid = [ map(lambda x: idx2label[x],
                                 rnn.classify(numpy.asarray(contextwin(x_src, 
				 model_options['win'])).astype('int32'),
                                 numpy.asarray(contextwin(x_tgt,model_options['win'])).astype('int32')))
                                 for x_src, x_tgt in zip(valid_s, valid_t) ]
            		groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
            		#words_valid = [ map(lambda x: idx2word_de[x], w) for w in valid_lex]


		elif model_options['use_char']:
			predictions_test = [ map(lambda x: idx2label[x],
				rnn.classify(numpy.asarray(contextwin(x, 
				model_options['win'])).astype('int32'),
				numpy.asarray(add_padding(_x, 
				model_options['max_char'])).astype('int32')))
                                for x, _x, in zip(test_t, test_tchar) ]
                	groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
                	#words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

                	predictions_valid = [ map(lambda x: idx2label[x],
				rnn.classify(numpy.asarray(contextwin(x, 
				model_options['win'])).astype('int32'),
				numpy.asarray(add_padding(_x,
				model_options['max_char'])).astype('int32')))
                                for x, _x, in zip(valid_t, valid_tchar) ]
                	groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
		else:
                	#evaluation // back into the real world : idx -> words
                	predictions_test = [ map(lambda x: idx2label[x],
				rnn.classify(numpy.asarray(contextwin(x,
				model_options['win'])).astype('int32'))) for x in test_t ]

                	groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
                	#words_test = [ map(lambda x: idx2word[x], w) for w in test_t]

                	predictions_valid = [ map(lambda x: idx2label[x], 
				rnn.classify(numpy.asarray(contextwin(x, 
				model_options['win'])).astype('int32'))) for x in valid_t ]
                	groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
                	#words_valid = [ map(lambda x: idx2word[x], w) for w in valid_t]

                #evaluation // compute the accuracy using conlleval.pl
		res_test = []
		res_valid = []
		current_score = 0
		if model_options['use_quest']:
                   res_test=wmt_eval(predictions_test, groundtruth_test, folder+'/current.test.txt')
               	   res_valid=wmt_eval(predictions_valid, groundtruth_valid, folder+'/current.valid.txt')
		   current_score = res_valid[2][0]
		if model_options['use_tag']:
                  res_test=icon_eval(predictions_test, groundtruth_test, folder+'/current.test.txt')
                  res_valid=icon_eval(predictions_valid, groundtruth_valid, folder+'/current.valid.txt')
		  current_score = res_valid[1]

                if current_score > best_f1:

		    """
			Save the model and model parameters
		    """
                    rnn.save(folder)
		    filename = folder +'/model'
		    with open('%s.json'%filename, 'wb') as f:
			  json.dump(model_options, f, indent=2)

                    best_f1 = current_score
                    if model_options['verbose']:
                        print 'NEW BEST: epoch', e, 'valid F1', res_valid, 'test F1' , res_test , ' '*20
                    model_options['be'] = e
		    subprocess.call(['mv', folder + '/current.test.txt.hyp', folder+'/best.test.txt'])
                    subprocess.call(['mv', folder + '/current.valid.txt.hyp', folder+'/best.valid.txt'])
                else:
                    print ''
          #Break if no improvement in 10 epochs
          if abs(model_options['be']-model_options['ce']) >= 10:  break
        print 'BEST RESULT: epoch', model_options['be'] , 'valid F1', best_f1 , 'with the model', folder

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	data = parser.add_argument_group('data sets; model loading and saving')
	data.add_argument('--use_model', type=str, required=True, metavar='PATH', nargs=1,
                         help="model name; GRU, LSTM, DeepLSTM, RNN (default GRU)")
	data.add_argument('--load_data', type=str, required=False, metavar='PATH', nargs=1,
                         help="path to the processed data file")
	data.add_argument('--data_train', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel training corpus (source, target and alignment)")
	data.add_argument('--data_train_y', type=str, required=True, metavar='PATH', nargs=1,
                         help="training labels")
	data.add_argument('--data_test', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel test corpus (source, target and alignment)")
	data.add_argument('--data_test_y', type=str, required=True, metavar='PATH', nargs=1,
                         help="test labels")
	data.add_argument('--data_valid', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel validation corpus (source, target and alignment)")
	data.add_argument('--data_valid_y', type=str, required=True, metavar='PATH', nargs=1,
                         help="validation labels")
	data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (source and target vocabulary)")
	data.add_argument('--character2index', type=str, required=True, metavar='PATH', nargs="+",
                         help="character vocabularies (source and target vocabulary)")
	data.add_argument('--label2index', type=str, required=True, metavar='PATH', nargs=1,
                         help="target labels to index dictionary")
	data.add_argument('--embeddings', type=str, metavar='PATH', nargs="+",
                         help="network vocabularies (source and target vocabulary)")
	data.add_argument('--use_quest', action="store_true",
                         help="use for quality estimation (default: %(default)s)")
	data.add_argument('--use_tag', action="store_true",
                         help="use for tagging task (default: %(default)s)")
	data.add_argument('--use_bilingual', action="store_true",
                         help="use bilingual model (default: %(default)s)")
	data.add_argument('--use_char', action="store_true",
                         help="use character as an additional feature(default: %(default)s)")
	data.add_argument('--use_pretrain', action="store_true",
                         help="use pretarining (default: %(default)s)")
	data.add_argument('--use_adadelta', action="store_true",
                         help="use adaptive learning rate (default: %(default)s)")
	data.add_argument('--saveto', action="store_true",
                         help="use adaptive learning rate (default: %(default)s)")
	args = parser.parse_args()

	#print vars(args)
	train(**vars(args))
