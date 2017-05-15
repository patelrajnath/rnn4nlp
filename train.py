#!/usr/bin/env python

import os
import sys
import codecs
import gzip
import cPickle
import time
import argparse
import  subprocess
import numpy 

from collections import OrderedDict
from rnn import GRU
from utils.tools import minibatch, contextwin, shuffle
from utils.data_preprocess import preprocess_data
from metrics.qe_eval import wmt_eval
from metrics.pos_eval import icon_eval

select_model = {"GRU_adadelta_bilingual_pretrain": GRU.GRU_adadelta_bilingual_pretrain,
		"GRU_adadelta_bilingual": GRU.GRU_adadelta_bilingual, 
		"GRU_adadelta_pretrain": GRU.GRU_adadelta_pretrain,
		"GRU_adadelta": GRU.GRU_adadelta,
		"GRU": GRU.GRU}

def train(dim_word=100,  # word vector dimensionality
          dim=100,  # the number of LSTM units
	  win=5, #Window size
	  bs=5, #number of backprop through time steps
	  seed=123,
	  verbose=1,
          use_model='GRU', #Choose the model from- LSTM, DEEPLSTM, RNN, 
          patience=10,  # early stopping patience
          max_epochs=50,
          lrate=0.0001,  # learning rate
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
          embeddings=['data/qe/pretrain/ep_qe.en.vector.txt',
              'data/qe/pretrain/ep_qe.de.vector.txt'],
	  use_adadelta=True,
          use_bilingual=False,
          use_pretrain=False,
          use_quest=False,
          use_tag=False,
          shuffle_each_epoch=True,
    ):

	folder = os.path.basename(__file__).split('.')[0]
    	if not os.path.exists(folder): os.mkdir(folder)

	model_options = OrderedDict(sorted(locals().copy().items()))

	print 'Model_Options:', model_options

	model_name = model_options['use_model']
	if use_adadelta:
		model_name += '_adadelta'
	if use_bilingual:
		model_name += '_bilingual'
	if use_pretrain:
		model_name += '_pretrain'

	print 'Using model:', model_name

	processed_data = preprocess_data(data_train=model_options['data_train'], 
		data_train_y=model_options['data_train_y'][0],
		data_valid=model_options['data_valid'], data_valid_y=model_options['data_valid_y'][0], 
		data_test=model_options['data_test'], data_test_y=model_options['data_test_y'][0], 
		dictionaries=model_options['dictionaries'],
		embeddings = model_options['embeddings'],
		use_bilingual=model_options['use_bilingual'], 
		use_pretrain=model_options['use_pretrain'])

	train, train_y, test, test_y, valid, valid_y, w2idxs, label2idxs, embs = processed_data
	idx2label = dict((k,v) for v,k in label2idxs.iteritems())

	vocsize_s = vocsize_t = 0
        emb_s, emb_t, train_s, train_t, test_s, test_t, valid_s, valid_t = ([] for i in range(8))
		
	if use_bilingual or len(train) == 2:
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

	#print test_t[0], test_s[0], test_y[0]

    	nclasses = len(label2idxs)
    	nsentences = len(train_t)

	print nsentences
	
    	numpy.random.seed(model_options['seed'])
    	# instanciate the model
    	rnn = select_model[model_name]( nh = model_options['dim'],
                    nc = nclasses,
                    de = model_options['dim_word'],
                    cs = model_options['win'],
		    ne_src = vocsize_s,
		    ne_tgt = vocsize_t,
		    emb_src = emb_s,
		    emb_tgt = emb_t )

    	# train with early stopping on validation set
    	best_f1 = -numpy.inf
    	model_options['patience'] = 2
    	batch_size = 100
    	n_batches = nsentences//batch_size
    	print n_batches
    	for e in xrange(model_options['max_epochs']):
	  model_options['ce'] = e
      	  #shuffle
	  if shuffle_each_epoch:
      	  	shuffle([train_t, train_s, train_y], model_options['seed'])

      	  tic = time.time()
      	  for k in xrange(n_batches):
            #Creating batches
	    batch_train_s = []
	    if use_bilingual:
            	batch_train_s = train_s[k*batch_size:(k+1)*batch_size]

            batch_train_t = train_t[k*batch_size:(k+1)*batch_size]
            batch_train_y = train_y[k*batch_size:(k+1)*batch_size]
            batch_err = 0
            for i in xrange(batch_size):
		cwords_src = []
		if model_options['use_bilingual']:
                	cwords_src = contextwin(batch_train_s[i], model_options['win'])

                cwords_tgt = contextwin(batch_train_t[i], model_options['win'])
                labels = batch_train_y[i]

		if model_options['use_bilingual']:
                     err = rnn.train_grad_shared(cwords_src, cwords_tgt, labels, model_options['lrate'])
		else:
                     err = rnn.train_grad_shared(cwords_tgt, labels, model_options['lrate'])

                rnn.train_update(model_options['lrate'])
                rnn.normalize()
                
                if model_options['verbose']:
                    print '[learning] epoch %i batch %i >> %2.2f%%'%(e, k, (i+1)*100./batch_size),'completed in %.2f (sec) <<\r'%(time.time()-tic),
		    sys.stdout.flush()

	    if(k % model_options['patience'] == 0):

		predictions_test, groundtruth_test, predictions_valid, \
			groundtruth_valid = ([] for i in range(4))

		if model_options['use_bilingual']:
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
                    rnn.save(folder)
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
	data.add_argument('--use_quest', action="store_true",
                         help="use for quality estimation (default: %(default)s)")
	data.add_argument('--use_tag', action="store_true",
                         help="use for tagging task (default: %(default)s)")
	data.add_argument('--use_bilingual', action="store_true",
                         help="use bilingual model (default: %(default)s)")
	data.add_argument('--use_pretrain', action="store_true",
                         help="use pretarining (default: %(default)s)")
	data.add_argument('--use_adadelta', action="store_false",
                         help="use adaptive learning rate (default: %(default)s)")
	data.add_argument('--data_train', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel training corpus (source, target and alignment)")
	data.add_argument('--data_train_y', type=str, required=True, metavar='PATH', nargs=1,
                         help="training labels")
	data.add_argument('--data_test', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel training corpus (source, target and alignment)")
	data.add_argument('--data_test_y', type=str, required=True, metavar='PATH', nargs=1,
                         help="training labels")
	data.add_argument('--data_valid', type=str, required=True, metavar='PATH', nargs="+",
                         help="parallel training corpus (source, target and alignment)")
	data.add_argument('--data_valid_y', type=str, required=True, metavar='PATH', nargs=1,
                         help="training labels")
	data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (source and target vocabulary)")
	data.add_argument('--embeddings', type=str, metavar='PATH', nargs="+",
                         help="network vocabularies (source and target vocabulary)")
	args = parser.parse_args()

	#print vars(args)
	train(**vars(args))
