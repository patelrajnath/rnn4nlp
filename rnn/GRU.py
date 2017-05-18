#!/usr/bin/env python

import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict
from utils.tools import numpy_floatX

class GRU(object):
    
    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None, 
		ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne_tgt+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        #Input layer weghts
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
              #H(t-1) weights
        self.Ur = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Uu = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.Wr, self.Wu, self.W, self.Ur, self.Uu,\
                       self.bh, self.br, self.bu, self.b, self.h0 ]
        
        self.names  = ['emb', 'Wx', 'Wh', 'Wr', 'Wu', 'W', 'Ur', 'Uu', 'bh', 'br', 'bu', 'b', 'h0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label
        
        def recurrence(x_t, h_tm1):
            #Gates
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(h_tm1, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(h_tm1, self.Uu) + self.bu)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((h_tm1*r), self.Wh) + self.bh)
            
            c = (1-z)*h + z*h_tm1
            
            s_t = T.nnet.softmax(T.dot(c, self.W) + self.b)
            
            return [c, s_t]

        [c, s_t],_ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])
        
        p_y_given_x_lastword = s_t[-1,0,:]
        p_y_given_x_sentence = s_t[:,0,:]
        
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates)

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class GRU_adadelta(object):

    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None, 
		ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne_tgt+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        #Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
              #H(t-1) weights
        self.Ur = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.Wr, self.Wu, self.W, self.Ur, self.Uu,\
                       self.bh, self.br, self.bu, self.b, self.h0 ]
        
        self.names  = ['emb', 'Wx', 'Wh', 'Wr', 'Wu', 'W', 'Ur', 'Uu', 'bh', 'br', 'bu', 'b', 'h0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.ivector('y') # label
        
        def recurrence(x_t, h_tm1):
            #Gates
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(h_tm1, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(h_tm1, self.Uu) + self.bu)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((h_tm1*r), self.Wh) + self.bh)
            
            c = (1-z)*h + z*h_tm1
            
            s_t = T.nnet.softmax(T.dot(c, self.W) + self.b)
            
            return [c, s_t]

        [c, s_t],_ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()

        grads = T.grad( nll, self.params )
        
	'''
        adadelta learning
	'''
        tparams = OrderedDict((p,q) for p,q in zip(self.names, self.params))
        
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        # theano functions
        self.train_grad_shared = theano.function(inputs=[idxs, y, lr], outputs=nll, updates=zgup + rg2up, 
					on_unused_input='ignore', name='adadelta_train_grad_shared', 
					allow_input_downcast=True)
        self.train_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_train_update')
        
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class GRU_adadelta_pretrain(object):
    
    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None, 
		ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        #self.emb = theano.shared(numpy.random.normal(0.0, 0.01,\
        #         (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.emb = theano.shared(numpy.asarray(emb).astype(theano.config.floatX))                 
	#Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
              #H(t-1) weights
        self.Ur = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.Wr, self.Wu, self.W, self.Ur, self.Uu,\
                       self.bh, self.br, self.bu, self.b, self.h0 ]
        
        self.names  = ['emb', 'Wx', 'Wh', 'Wr', 'Wu', 'W', 'Ur', 'Uu', 'bh', 'br', 'bu', 'b', 'h0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.ivector('y') # label
        
        def recurrence(x_t, h_tm1):
            #Gates
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(h_tm1, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(h_tm1, self.Uu) + self.bu)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((h_tm1*r), self.Wh) + self.bh)
            
            c = (1-z)*h + z*h_tm1
            
            s_t = T.nnet.softmax(T.dot(c, self.W) + self.b)
            
            return [c, s_t]

        [c, s_t],_ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()

        grads = T.grad( nll, self.params )
        
	'''
        adadelta learning
	'''
        tparams = OrderedDict((p,q) for p,q in zip(self.names, self.params))
        
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        # theano functions
        self.train_grad_shared = theano.function(inputs=[idxs, y, lr], outputs=nll, updates=zgup + rg2up, 
					on_unused_input='ignore', name='adadelta_train_grad_shared', 
					allow_input_downcast=True)
        self.train_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_train_update')
        
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


class GRU_adadelta_bilingual(object):
    
    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None, 
		ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb_src = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne_src+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.emb_tgt = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne_tgt+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        normalization_list = [self.emb_src, self.emb_tgt]
        #Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
              #H(t-1) weights
        self.Ur = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb_src, self.emb_tgt,self.Wx, self.Wh, self.Wr, self.Wu, self.W, self.Ur, self.Uu,\
                       self.bh, self.br, self.bu, self.b, self.h0 ]
        
        self.names  = ['emb_src', 'emb_tgt', 'Wx', 'Wh', 'Wr', 'Wu', 'W', 'Ur', 'Uu', 'bh', 'br', 'bu', 'b', 'h0']
        
        idxs_src = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        idxs_tgt = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        y    = T.ivector('y') # label
        
        def recurrence(x1, x2, h_tm1):
            #Gates
	    x_t = T.concatenate([x1, x2])
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(h_tm1, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(h_tm1, self.Uu) + self.bu)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((h_tm1*r), self.Wh) + self.bh)
            
            c = (1-z)*h + z*h_tm1
            
            s_t = T.nnet.softmax(T.dot(c, self.W) + self.b)
            
            return [c, s_t]

        [c, s_t],_ = theano.scan(fn=recurrence, \
            sequences=[x_src, x_tgt], outputs_info=[self.h0, None], \
            n_steps=x_src.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()

        grads = T.grad( nll, self.params )
        
	'''
        adadelta learning
	'''
        tparams = OrderedDict((p,q) for p,q in zip(self.names, self.params))
        
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        # theano functions
        self.train_grad_shared = theano.function(inputs=[idxs_src, idxs_tgt, y, lr], outputs=nll, updates=zgup + rg2up, 
					on_unused_input='ignore', name='adadelta_train_grad_shared', 
					allow_input_downcast=True)
        self.train_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_train_update')
        
        self.classify = theano.function(inputs=[idxs_src, idxs_tgt], outputs=y_pred)

        self.normalize = theano.function( inputs = [],
                         updates = OrderedDict((emb, emb/T.sqrt((emb**2).sum(axis=1)).dimshuffle(0,'x'))\
                                               for emb in normalization_list))
        
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


class GRU_adadelta_bilingual_pretrain(object):

    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None, 
		ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb_tgt = theano.shared(numpy.asarray(emb_tgt).astype(theano.config.floatX)) 
        self.emb_src = theano.shared(numpy.asarray(emb_src).astype(theano.config.floatX)) 
       
	normalization_list = [self.emb_src, self.emb_tgt]
        #Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
              #H(t-1) weights
        self.Ur = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb_src, self.emb_tgt,self.Wx, self.Wh, self.Wr, self.Wu, self.W, self.Ur, self.Uu,\
                       self.bh, self.br, self.bu, self.b, self.h0 ]
        
        self.names  = ['emb_src', 'emb_tgt', 'Wx', 'Wh', 'Wr', 'Wu', 'W', 'Ur', 'Uu', 'bh', 'br', 'bu', 'b', 'h0']
        
        idxs_src = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        idxs_tgt = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        y    = T.ivector('y') # label
        
        def recurrence(x1, x2, h_tm1):
            #Gates
	    x_t = T.concatenate([x1, x2])
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(h_tm1, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(h_tm1, self.Uu) + self.bu)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((h_tm1*r), self.Wh) + self.bh)
            
            c = (1-z)*h + z*h_tm1
            
            s_t = T.nnet.softmax(T.dot(c, self.W) + self.b)
            
            return [c, s_t]

        [c, s_t],_ = theano.scan(fn=recurrence, \
            sequences=[x_src, x_tgt], outputs_info=[self.h0, None], \
            n_steps=x_src.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()

        grads = T.grad( nll, self.params )
        
	'''
        adadelta learning
	'''
        tparams = OrderedDict((p,q) for p,q in zip(self.names, self.params))
        
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        # theano functions
        self.train_grad_shared = theano.function(inputs=[idxs_src, idxs_tgt, y, lr], outputs=nll, updates=zgup + rg2up, 
					on_unused_input='ignore', name='adadelta_train_grad_shared', 
					allow_input_downcast=True)
        self.train_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_train_update')
        
        self.classify = theano.function(inputs=[idxs_src, idxs_tgt], outputs=y_pred)

        self.normalize = theano.function( inputs = [],
                         updates = OrderedDict((emb, emb/T.sqrt((emb**2).sum(axis=1)).dimshuffle(0,'x'))			 for emb in normalization_list))

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class GRU_adadelta_char(object):
    
    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None,
                ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne_tgt + 1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
	#self.emb =  theano.shared(numpy.asarray(emb_tgt).astype(theano.config.floatX))    
	normalization_list = [self.emb]

        self.emb_char = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne_char, de_char)).astype(theano.config.floatX)) # add one for PADDING at the end
        #Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wx_char  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de_char * max_char, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.Wh_char  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh_char  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wr_char = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de_char * max_char, nh)).astype(theano.config.floatX))
        self.Wu_char = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de_char * max_char, nh)).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ur = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Ur_char = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu_char = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.br_char   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu_char   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.W_char   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b_char   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.emb_char, self.Wx, self.Wx_char, self.Wh, \
	self.Wh_char, self.Wr, self.Wr_char, self.Wu, self.Wu_char, self.W, \
	self.W_char, self.Ur, self.Ur_char, self.Uu, self.Uu_char, self.bh, \
	self.bh_char, self.br, self.br_char, self.bu, self.bu_char, self.b, self.b_char, self.h0 ]
        
        self.names  = ['emb', 'emb_char','Wx', 'Wx_char', 'Wh', 'Wh_char', 'Wr',\
	 'Wr_char', 'Wu','Wu_char', 'W', 'W_char', 'Ur', 'Ur_char', 'Uu','Uu_char',\
	 'bh', 'bh_char', 'br','br_char', 'bu', 'bu_char', 'b', 'b_char', 'h0']
        
        idxs = T.imatrix('y') #as many columns as context window size/lines as words in the sentence
        idxs_char = T.imatrix('c')

        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        x_char = self.emb_char[idxs_char].reshape((idxs_char.shape[0], de_char*max_char))
        y    = T.ivector('y') # label
        
        def recurrence(x_t, _x, _h, _ch, _s):
            #Gates
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(_h, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(_h, self.Uu) + self.bu)

            r_char = T.nnet.sigmoid(T.dot(_x, self.Wr_char) + T.dot(_ch, self.Ur_char) + self.br_char)
            z_char = T.nnet.sigmoid(T.dot(_x, self.Wu_char) + T.dot(_ch, self.Uu_char) + self.bu_char)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((_h*r), self.Wh) + self.bh)
            h_char = T.tanh(T.dot(_x, self.Wx_char) + T.dot((_ch*r_char), self.Wh_char) + self.bh_char)
            
            c = (1-z)*h + z*_h
            c_char = (1-z_char)*h_char + z_char*_ch
            
            s = T.nnet.sigmoid(T.dot(c, self.W) + T.dot(c_char, self.W_char) + self.b_char)
            s_t = T.nnet.softmax(T.dot(c, self.W) + T.dot(c_char, self.W_char) + _s + self.b)
            
            return [c, c_char, s, s_t]

        [c, ch, s, s_t],_ = theano.scan(fn=recurrence, \
            sequences=[x, x_char], outputs_info=[self.h0, T.alloc(numpy_floatX(0.),nh),
					T.alloc(numpy_floatX(0.), nc), None],
					n_steps=x.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()

        grads = T.grad( nll, self.params )
        
	'''
        adadelta learning
	'''
        tparams = OrderedDict((p,q) for p,q in zip(self.names, self.params))
        
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        # theano functions
        self.train_grad_shared = theano.function(inputs=[idxs, idxs_char, y, lr],
				outputs=nll, updates=zgup + rg2up, 
				on_unused_input='ignore', name='adadelta_train_grad_shared', 
				allow_input_downcast=True)
        self.train_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_train_update')
        
        self.classify = theano.function(inputs=[idxs, idxs_char], outputs=y_pred)

        self.normalize = theano.function( inputs = [],
                         updates = OrderedDict((emb, emb/T.sqrt((emb**2).sum(axis=1)).dimshuffle(0,'x'))                         for emb in normalization_list)) 
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class GRU_adadelta_char_bilingual_pretrain(object):
    def __init__(self, nh, nc, de, cs, ne_char=None, de_char=None, ne_src=None, 
		ne_tgt=None, emb_src=None, emb_tgt=None, max_char=10):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
	self.emb_src =  theano.shared(numpy.asarray(emb_src).astype(theano.config.floatX))    
	self.emb_tgt =  theano.shared(numpy.asarray(emb_tgt).astype(theano.config.floatX))    
	normalization_list = [self.emb_src, self.emb_tgt]

        self.emb_char = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne_char, de_char)).astype(theano.config.floatX))
        #Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wx_char  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de_char * max_char, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.Wh_char  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh_char  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wr_char = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de_char * max_char, nh)).astype(theano.config.floatX))
        self.Wu_char = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de_char * max_char, nh)).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ur = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Ur_char = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uu_char = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.br_char   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bu_char   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.W_char   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b_char   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb_src, self.emb_tgt, self.emb_char, self.Wx, self.Wx_char, self.Wh, \
	self.Wh_char, self.Wr, self.Wr_char, self.Wu, self.Wu_char, self.W, \
	self.W_char, self.Ur, self.Ur_char, self.Uu, self.Uu_char, self.bh, \
	self.bh_char, self.br, self.br_char, self.bu, self.bu_char, self.b, self.b_char, self.h0 ]
        
        self.names  = ['emb_src', 'emb_tgt', 'emb_char','Wx', 'Wx_char', 'Wh', 'Wh_char', 'Wr',\
	 'Wr_char', 'Wu','Wu_char', 'W', 'W_char', 'Ur', 'Ur_char', 'Uu','Uu_char',\
	 'bh', 'bh_char', 'br','br_char', 'bu', 'bu_char', 'b', 'b_char', 'h0']
        
        idxs_src = T.imatrix('x') # as many columns as context window size/lines as words in the sentence
        idxs_tgt = T.imatrix('y') # as many columns as context window size/lines as words in the sentence
        idxs_char = T.imatrix('c')

        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        x_char = self.emb_char[idxs_char].reshape((idxs_char.shape[0], de_char*max_char))
        y    = T.ivector('y') # label
        
        def recurrence(x1, x2, _x, _h, _ch, _s):
	    x_t = T.concatenate([x1, x2])
            #Gates
            r = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(_h, self.Ur) + self.br)
            z = T.nnet.sigmoid(T.dot(x_t, self.Wu) + T.dot(_h, self.Uu) + self.bu)

            r_char = T.nnet.sigmoid(T.dot(_x, self.Wr_char) + T.dot(_ch, self.Ur_char) + self.br_char)
            z_char = T.nnet.sigmoid(T.dot(_x, self.Wu_char) + T.dot(_ch, self.Uu_char) + self.bu_char)
            
            h = T.tanh(T.dot(x_t, self.Wx) + T.dot((_h*r), self.Wh) + self.bh)
            h_char = T.tanh(T.dot(_x, self.Wx_char) + T.dot((_ch*r_char), self.Wh_char) + self.bh_char)
            
            c = (1-z)*h + z*_h
            c_char = (1-z_char)*h_char + z_char*_ch
            
            s = T.nnet.sigmoid(T.dot(c, self.W) + T.dot(c_char, self.W_char) + self.b_char)
            s_t = T.nnet.softmax(T.dot(c, self.W) + T.dot(c_char, self.W_char) + _s + self.b)
            
            return [c, c_char, s, s_t]

        [c, ch, s, s_t],_ = theano.scan(fn=recurrence, \
            sequences=[x_src, x_tgt, x_char], outputs_info=[self.h0, T.alloc(numpy_floatX(0.),nh),
					T.alloc(numpy_floatX(0.), nc), None],
					n_steps=x_src.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()

        grads = T.grad( nll, self.params )
        
	'''
        adadelta learning
	'''
        tparams = OrderedDict((p,q) for p,q in zip(self.names, self.params))
        
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        
        # theano functions
        self.train_grad_shared = theano.function(inputs=[idxs_src, idxs_tgt, idxs_char, y, lr], outputs=nll, updates=zgup + rg2up, 
					on_unused_input='ignore', name='adadelta_train_grad_shared', 
					allow_input_downcast=True)
        self.train_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_train_update')
        
        self.classify = theano.function(inputs=[idxs_src, idxs_tgt, idxs_char], outputs=y_pred)

        self.normalize = theano.function( inputs = [],
                         updates = OrderedDict((emb, emb/T.sqrt((emb**2).sum(axis=1)).dimshuffle(0,'x'))                         for emb in normalization_list)) 
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
