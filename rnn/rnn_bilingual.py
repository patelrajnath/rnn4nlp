import theano
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

class model_with_pretraining(object):
    
    def __init__(self, nh, nc, ne, de, cs, emb_tgt, emb_src):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        #self.emb = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
        #           (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        
        self.emb_src = theano.shared(np.asarray(emb_src).astype(theano.config.floatX)) 
        self.emb_tgt = theano.shared(np.asarray(emb_tgt).astype(theano.config.floatX)) 

        normalization_list = [self.emb_src, self.emb_tgt]
        
        # add one for PADDING at the end
        
        self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (2*de * cs, nh)).astype(theano.config.floatX))

        #Recurent memory
        self.h0  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        #Reccurant weight
        self.Wh  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        
        self.W   = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(np.zeros(nc, dtype=theano.config.floatX))
       

        # bundle
        self.params = [ self.emb_src, self.emb_tgt, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['emb_src', 'emb_tgt','Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        
        idxs_src =  T.imatrix()
        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        idxs_tgt = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        
        y    = T.ivector('y') # label
        
        def recurrence(x_src, x_tgt, h_tm1):
            
            x_t = T.concatenate([x_src, x_tgt])
            
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=[x_src, x_tgt], outputs_info=[self.h0, None], \
            n_steps=x_src.shape[0], truncate_gradient=7)

        #p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s[:,0,:])[T.arange(y.shape[0]), y]).sum()
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs_src, idxs_tgt], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs_src, idxs_tgt, y, lr],
                                      outputs = [nll, s],
                                      updates = updates,
				      allow_input_downcast=True )

        self.normalize = theano.function( inputs = [],
                         updates = OrderedDict((emb, emb/T.sqrt((emb**2).sum(axis=1)).dimshuffle(0,'x'))\
                                               for emb in normalization_list))
 
        
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
