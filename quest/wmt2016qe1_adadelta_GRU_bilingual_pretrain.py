import numpy
import numpy as np
import time
import sys
import subprocess
import os
import random
import gzip
import cPickle
from metrics.qe_eval import wmt_eval
from utils.tools import minibatch, contextwin, shuffle
from rnn.GRU_adadelta_bilingual_pretrain import modelGRU_bilingual_pretrain

if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.00627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':5, # number of words in the context window
         'bs':5, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':100, # dimension of word embedding
         'nepochs':50}
  
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    data_train = 'data/pretrain_data6_wmt2016qe1.pkl.gz'
    f = gzip.open(data_train,'rb')
    train, test, valid, embeddings, dicts = cPickle.load(f)

    train_lex, train_s, train_y = train
    test_lex, test_s, test_y = test
    valid_lex, valid_s, valid_y = valid
    emb_src, emb_tgt = embeddings
    
    idx2label = dict((k,v) for v,k in dicts['label2index'].iteritems())
    idx2word_en  = dict((k,v) for v,k in dicts['word2index_en'].iteritems())
    idx2word_de  = dict((k,v) for v,k in dicts['word2index_de'].iteritems())

    #print len(dicts['word2index_de'])

    vocsize_de = len(dicts['word2index_de'])
    vocsize_en = len(dicts['word2index_en'])
    nclasses = len(dicts['label2index'])
    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = modelGRU_bilingual_pretrain( nh = s['nhidden'],
                    nc = nclasses,
                    de = s['emb_dimension'],
                    cs = s['win'],
		    emb_src = emb_src,
		    emb_tgt = emb_tgt )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    s['patience'] = 2
    batch_size = 1000
    n_batches = nsentences//batch_size
    print n_batches
    for e in xrange(s['nepochs']):
      # shuffle
      shuffle([train_lex, train_s, train_y], s['seed'])
      s['ce'] = e
      tic = time.time()
      for k in xrange(n_batches):
            #Creating batches
            batch_train_s = train_s[k*batch_size:(k+1)*batch_size]
            batch_train_lex = train_lex[k*batch_size:(k+1)*batch_size]
            batch_train_y = train_y[k*batch_size:(k+1)*batch_size]
            batch_err = 0
            for i in xrange(batch_size):
                cwords_src = contextwin(batch_train_s[i], s['win'])
                cwords_tgt = contextwin(batch_train_lex[i], s['win'])
                labels = batch_train_y[i]
                err = rnn.train_grad_shared(cwords_src, cwords_tgt, labels, s['clr'])
                rnn.train_update(s['clr'])
                rnn.normalize()
                
                if s['verbose']:
                    print '[learning] epoch %i batch %i >> %2.2f%%'%(e, k, (i+1)*100./batch_size),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stdout.flush()
      	    if(k % s['patience'] == 0):
 		predictions_test = [ map(lambda x: idx2label[x], \
                                 rnn.classify(numpy.asarray(contextwin(x_src, s['win'])).astype('int32'),\
                                 numpy.asarray(contextwin(x_tgt, s['win'])).astype('int32')))\
                                 for x_src, x_tgt in zip(test_s, test_lex) ]
            	groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
            	#words_test = [ map(lambda x: idx2word_de[x], w) for w in test_lex]

            	predictions_valid = [ map(lambda x: idx2label[x], \
                                 rnn.classify(numpy.asarray(contextwin(x_src, s['win'])).astype('int32'), \
                                 numpy.asarray(contextwin(x_tgt, s['win'])).astype('int32')))\
                                 for x_src, x_tgt in zip(valid_s, valid_lex) ]
            	groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
            	#words_test = [ map(lambda x: idx2word_de[x], w) for w in valid_lex]

                # evaluation // compute the accuracy using conlleval.pl
                res_test  = wmt_eval(predictions_test, groundtruth_test, folder + '/current.test.txt')
                res_valid = wmt_eval(predictions_valid, groundtruth_valid, folder + '/current.valid.txt')

                if res_valid[2][0] > best_f1:
                    rnn.save(folder)
                    best_f1 = res_valid[2][0]
                    if s['verbose']:
                        print 'NEW BEST: epoch', e, 'valid F1', res_valid, 'test F1' , res_test , ' '*20
                    s['be'] = e
		    subprocess.call(['mv', folder + '/current.test.txt.hyp', folder + '/best.test.txt'])
                    subprocess.call(['mv', folder + '/current.valid.txt.hyp', folder + '/best.valid.txt'])
                else:
                    print ''
       # Break if no improvement in 10 epochs
      if abs(s['be']-s['ce']) >= 10:  break
    print 'BEST RESULT: epoch', s['be'] , 'valid F1', best_f1 , 'with the model', folder


