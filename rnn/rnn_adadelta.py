import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict
from qe_rnn.utils.tools import numpy_floatX

class model(object):
    
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (de * cs, nh)).astype(theano.config.floatX))

        #Recurent memory
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        #Reccurant weight
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['emb', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y = T.ivector('y') # label
        
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]        

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        #p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s[:,0,:])[T.arange(y.shape[0]), y]).sum()
        
        grads = T.grad( nll, self.params )
        
        '''
        Adadelta update
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
                                    on_unused_input='ignore', name='adadelta_train_grad_shared', allow_input_downcast=True)
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
