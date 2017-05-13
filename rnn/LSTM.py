import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class modeLSTM(object):
    
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
        
        #LSTM Gate
        #X(t) weights
        self.Wi = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wf = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ui = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Uo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Uf = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        #Bias
        self.bi   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bo   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bf   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.c0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Wo, self.Wf, self.W, self.Ui, self.Uo, self.Uf, \
                       self.bh, self.bi, self.bo, self.bf, self.b, self.h0 ]
        #self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Wf, self.W, self.Ui, self.Uo, self.Uf, \
        #               self.bh, self.bi, self.bo, self.bf, self.b, self.h0 ]
        #self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Ui, self.W,\
        #               self.bh, self.bi, self.b, self.h0 ]
        
        self.names  = ['embeddings', 'Wx', 'Wh', 'Wi', 'Wo', 'Wf', 'W', 'Ui', 'Uo', 'Uf', 'bh', 'bi', 'bo', 'bf', 'b', 'h0', 'c0']
        #self.names  = ['embeddings', 'Wx', 'Wh', 'Wi', 'W', 'bh', 'bi', 'b', 'h0', 'c0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label
        
        def recurrence(x_t, h_tm1, c_):
            #Gates
            g_i = T.nnet.sigmoid(T.dot(x_t, self.Wi) + T.dot(h_tm1, self.Ui) + self.bi)
            g_o = T.nnet.sigmoid(T.dot(x_t, self.Wo) + T.dot(h_tm1, self.Uo) + self.bo)
            g_f = T.nnet.sigmoid(T.dot(x_t, self.Wf) + T.dot(h_tm1, self.Uf) + self.bf)
            
            g_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            
            c = g_f*c_ + g_i*g_t
            #c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = g_o*T.tanh(c)
            #h = m_[:, None] * h + (1. - m_)[:, None] * h_tm1
            
            s_t = T.nnet.softmax(T.dot(h, self.W) + self.b)
            return [h, c, s_t]

        [h, c, s_t],_ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, self.c0, None], \
            n_steps=x.shape[0])

 
        
        p_y_given_x_lastword = s_t[-1,0,:]
        p_y_given_x_sentence = s_t[:,0,:]
        
        #p_y_given_x_lastword = s[-1]
        #p_y_given_x_sentence = s
        
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
