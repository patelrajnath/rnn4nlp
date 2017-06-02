import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class LSTM(object):
    
    def __init__(self, params, de, cs, de_char=None, max_char=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        self.emb = theano.shared(numpy.asarray(params['emb']).astype(theano.config.floatX)) # add one for PADDING at the end
        #Input layer weghts
        self.Wx  = theano.shared(numpy.asarray(params['Wx']).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.asarray(params['Wh']).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.asarray(params['bh']).astype(theano.config.floatX))
        
        #LSTM Gate
        #X(t) weights
        self.Wi = theano.shared(numpy.asarray(params['Wi']).astype(theano.config.floatX))
        self.Wo = theano.shared(numpy.asarray(params['Wo']).astype(theano.config.floatX))
        self.Wf = theano.shared(numpy.asarray(params['Wf']).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ui = theano.shared(numpy.asarray(params['Ui']).astype(theano.config.floatX))
        self.Uo = theano.shared(numpy.asarray(params['Uo']).astype(theano.config.floatX))
        self.Uf = theano.shared(numpy.asarray(params['Uf']).astype(theano.config.floatX))
        #Bias
        self.bi   = theano.shared(numpy.asarray(params['bi']).astype(theano.config.floatX))
        self.bo   = theano.shared(numpy.asarray(params['bo']).astype(theano.config.floatX))
        self.bf   = theano.shared(numpy.asarray(params['bf']).astype(theano.config.floatX))

        #output weights and biases
        self.W   = theano.shared(numpy.asarray(params['W']).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.asarray(params['b']).astype(theano.config.floatX))
        
        #Recurent memory
        self.h0  = theano.shared(numpy.asarray(params['h0']).astype(theano.config.floatX))
        self.c0  = theano.shared(numpy.asarray(params['c0']).astype(theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.Wi, self.Wo, self.Wf, self.W, self.Ui, self.Uo, self.Uf, \
                       self.bh, self.bi, self.bo, self.bf, self.b, self.h0, self.c0]
    
        self.names  = ['emb', 'Wx', 'Wh', 'Wi', 'Wo', 'Wf', 'W', 'Ui', 'Uo', 'Uf', 'bh', 'bi', 'bo', 'bf', 'b', 'h0', 'c0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.ivector('y') # label
        
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
        
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

class LSTM_bilingual(object):
    
    def __init__(self, params, de, cs, de_char=None, max_char=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        self.emb_src = theano.shared(numpy.asarray(params['emb_src']).astype(theano.config.floatX)) # add one for PADDING at the end
        self.emb_tgt = theano.shared(numpy.asarray(params['emb_tgt']).astype(theano.config.floatX)) # add one for PADDING at the end

        normalization_list = [self.emb_src, self.emb_tgt]

        #Input layer weghts
        self.Wx  = theano.shared(numpy.asarray(params['Wx']).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.asarray(params['Wh']).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.asarray(params['bh']).astype(theano.config.floatX))
        
        #LSTM Gate
        #X(t) weights
        self.Wi = theano.shared(numpy.asarray(params['Wi']).astype(theano.config.floatX))
        self.Wo = theano.shared(numpy.asarray(params['Wo']).astype(theano.config.floatX))
        self.Wf = theano.shared(numpy.asarray(params['Wf']).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ui = theano.shared(numpy.asarray(params['Ui']).astype(theano.config.floatX))
        self.Uo = theano.shared(numpy.asarray(params['Uo']).astype(theano.config.floatX))
        self.Uf = theano.shared(numpy.asarray(params['Uf']).astype(theano.config.floatX))
        #Bias
        self.bi   = theano.shared(numpy.asarray(params['bi']).astype(theano.config.floatX))
        self.bo   = theano.shared(numpy.asarray(params['bo']).astype(theano.config.floatX))
        self.bf   = theano.shared(numpy.asarray(params['bf']).astype(theano.config.floatX))

        #output weights and biases
        self.W   = theano.shared(numpy.asarray(params['W']).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.asarray(params['b']).astype(theano.config.floatX))
        
        #Recurent memory
        self.h0  = theano.shared(numpy.asarray(params['h0']).astype(theano.config.floatX))
        self.c0  = theano.shared(numpy.asarray(params['c0']).astype(theano.config.floatX))
        
        idxs_src = T.imatrix() # as many columns as context window size/lines as words in the sentence
        idxs_tgt = T.imatrix() # as many columns as context window size/lines as words in the sentence

        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))

        y    = T.ivector('y') # label
        
        def recurrence(x1, x2, h_tm1, c_):
            #Gates
            x_t = T.concatenate([x1, x2])
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
            sequences=[x_src, x_tgt], outputs_info=[self.h0, self.c0, None], \
            n_steps=x_src.shape[0])

        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # theano functions
        self.classify = theano.function(inputs=[idxs_src, idxs_tgt], outputs=y_pred)
