import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class modeLSTM_DEEP(object):
    
    def __init__(self, nh, nc, ne, de, cs, nh2=100):
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
        
        
        #Weights for hidden layer-2
        #Input layer weghts
        self.Wx2  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh2)).astype(theano.config.floatX))
        #Reccurant weight or g(t) in note
        self.Wh2  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh2, nh2)).astype(theano.config.floatX))
        self.bh2  = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        #LSTM Gate
        #X(t) weights
        self.Wi2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh2)).astype(theano.config.floatX))
        self.Wo2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh2)).astype(theano.config.floatX))
        self.Wf2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh2)).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ui2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh2, nh2)).astype(theano.config.floatX))
        self.Uo2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh2, nh2)).astype(theano.config.floatX))
        self.Uf2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh2, nh2)).astype(theano.config.floatX))
        #Bias
        self.bi2   = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        self.bo2   = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        self.bf2   = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))

        #output weights and biases
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh2, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX)) 
        
        #Recurent memory
        self.h1  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.c1  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
  
        self.h2  = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        self.c2  = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Wo, self.Wf, self.Wx2, self.Wh2, self.Wi2, self.Wo2, self.Wf2, self.W,\
                       self.Ui, self.Uo, self.Uf,  self.Ui2, self.Uo2, self.Uf2,\
                       self.bh, self.bi, self.bo, self.bf, self.bh2, self.bi2, self.bo2, self.bf2, self.b,\
                       self.h1, self.c1, self.h2, self.c2]
        #self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Wf, self.W, self.Ui, self.Uo, self.Uf, \
        #               self.bh, self.bi, self.bo, self.bf, self.b, self.h0 ]
        #self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Ui, self.W,\
        #               self.bh, self.bi, self.b, self.h0 ]
        
        self.names  = ['embeddings', 'Wx', 'Wh', 'Wi', 'Wo', 'Wf','Wx2', 'Wh2', 'Wi2', 'Wo2', 'Wf2', 'W', \
                       'Ui', 'Uo', 'Uf',  'Ui2', 'Uo2', 'Uf2',\
                       'bh', 'bi', 'bo', 'bf', 'bh2', 'bi2', 'bo2', 'bf2',\
                       'b', 'h1', 'c1', 'h2', 'c2']
        #self.names  = ['embeddings', 'Wx', 'Wh', 'Wi', 'W', 'bh', 'bi', 'b', 'h0', 'c0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label
        
        def recurrence(x_t, h_tm1, c_, h_tm2, c2_):
            #Gates
            g_i = T.nnet.sigmoid(T.dot(x_t, self.Wi) + T.dot(h_tm1, self.Ui) + self.bi)
            g_o = T.nnet.sigmoid(T.dot(x_t, self.Wo) + T.dot(h_tm1, self.Uo) + self.bo)
            g_f = T.nnet.sigmoid(T.dot(x_t, self.Wf) + T.dot(h_tm1, self.Uf) + self.bf)
            
            g_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            
            c1 = g_f*c_ + g_i*g_t
            #c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h1 = g_o*T.tanh(c1)
            #h = m_[:, None] * h + (1. - m_)[:, None] * h_tm1
            
            g_i2 = T.nnet.sigmoid(T.dot(h1, self.Wi2) + T.dot(h_tm2, self.Ui2) + self.bi2)
            g_o2 = T.nnet.sigmoid(T.dot(h1, self.Wo2) + T.dot(h_tm2, self.Uo2) + self.bo2)
            g_f2 = T.nnet.sigmoid(T.dot(h1, self.Wf2) + T.dot(h_tm2, self.Uf2) + self.bf2)
            
            g_t2 = T.tanh(T.dot(h1, self.Wx2) + T.dot(h_tm2, self.Wh2) + self.bh2)
            
            c2 = g_f2*c2_ + g_i2*g_t2
            #c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h2 = g_o2*T.tanh(c2)
            
            #h3 = T.concatenate([h1 , h2])
            s_t = T.nnet.softmax(T.dot(h2, self.W) + self.b)
            return [h1, c1, h2, c2, s_t]

        [h1, c1, h2, c2, s_t],_ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h1, self.c1, self.h2, self.c2, None], \
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
                                      outputs = [nll, s_t],
                                      updates = updates)

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
