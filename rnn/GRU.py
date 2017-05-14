import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class modelGRU(object):
    
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
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