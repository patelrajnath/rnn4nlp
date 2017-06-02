import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict
from utils.tools import numpy_floatX

class GRU(object):
    
    def __init__(self, params, de, cs, de_char=None, max_char=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(numpy.array(params['emb']).astype(theano.config.floatX))
        #Input layer weghts
        self.Wx  = theano.shared(numpy.array(params['Wx']).astype(theano.config.floatX))
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.array(params['Wh']).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.array(params['bh']).astype(theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.array(params['Wr']).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.array(params['Wu']).astype(theano.config.floatX))
        
        #H(t-1) weights
        self.Ur = theano.shared(numpy.array(params['Ur']).astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.array(params['Uu']).astype(theano.config.floatX))
        #Bias
        self.br = theano.shared(numpy.array(params['br']).astype(theano.config.floatX))
        self.bu = theano.shared(numpy.array(params['bu']).astype(theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.array(params['W']).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.array(params['b']).astype(theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.array(params['h0']).astype(theano.config.floatX))
        # bundle
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

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

class GRU_bilingual(object):
    
    def __init__(self, params, de, cs, de_char=None, max_char=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb_src = theano.shared(numpy.array(params['emb_src']).astype(theano.config.floatX))
        self.emb_tgt = theano.shared(numpy.array(params['emb_tgt']).astype(theano.config.floatX))

        normalization_list = [self.emb_src, self.emb_tgt]

        #Input layer weghts
        self.Wx  = theano.shared(numpy.array(params['Wx']).astype(theano.config.floatX))
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.array(params['Wh']).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.array(params['bh']).astype(theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.array(params['Wr']).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.array(params['Wu']).astype(theano.config.floatX))
        
        #H(t-1) weights
        self.Ur = theano.shared(numpy.array(params['Ur']).astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.array(params['Uu']).astype(theano.config.floatX))
        #Bias
        self.br = theano.shared(numpy.array(params['br']).astype(theano.config.floatX))
        self.bu = theano.shared(numpy.array(params['bu']).astype(theano.config.floatX))
        
        #output weights and biases
        self.W   = theano.shared(numpy.array(params['W']).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.array(params['b']).astype(theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.array(params['h0']).astype(theano.config.floatX))
        # bundle
        
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

        self.classify = theano.function(inputs=[idxs_src, idxs_tgt], outputs=y_pred)

class GRU_char(object):
    
    def __init__(self, params, de, cs, de_char=None, max_char=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb =  theano.shared(numpy.array(params['emb']).astype(theano.config.floatX))    
        normalization_list = [self.emb]

        self.emb_char = theano.shared(numpy.array(params['emb_char']).astype(theano.config.floatX)) 
        #Input layer weghts
        self.Wx  = theano.shared(numpy.array(params['Wx']).astype(theano.config.floatX))
        self.Wx_char  = theano.shared(numpy.array(params['Wx_char']).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.array(params['Wh']).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.array(params['bh']).astype(theano.config.floatX))
        self.Wh_char  = theano.shared(numpy.array(params['Wh_char']).astype(theano.config.floatX))
        self.bh_char  = theano.shared(numpy.array(params['bh_char']).astype(theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.array(params['Wr']).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.array(params['Wu']).astype(theano.config.floatX))
        self.Wr_char = theano.shared(numpy.array(params['Wr_char']).astype(theano.config.floatX))
        self.Wu_char = theano.shared(numpy.array(params['Wu_char']).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ur = theano.shared(numpy.array(params['Ur']).astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.array(params['Uu']).astype(theano.config.floatX))
        self.Ur_char = theano.shared(numpy.array(params['Ur_char']).astype(theano.config.floatX))
        self.Uu_char = theano.shared(numpy.array(params['Uu_char']).astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.array(params['br']).astype(theano.config.floatX))
        self.bu   = theano.shared(numpy.array(params['bu']).astype(theano.config.floatX))
        self.br_char   = theano.shared(numpy.array(params['br_char']).astype(theano.config.floatX))
        self.bu_char   = theano.shared(numpy.array(params['bu_char']).astype(theano.config.floatX))
    
        #output weights and biases
        self.W   = theano.shared(numpy.array(params['W']).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.array(params['b']).astype(theano.config.floatX))
        self.W_char   = theano.shared(numpy.array(params['W_char']).astype(theano.config.floatX))
        self.b_char   = theano.shared(numpy.array(params['b_char']).astype(theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.array(params['h0']).astype(theano.config.floatX))
        self.h0_char  = theano.shared(numpy.array(params['h0_char']).astype(theano.config.floatX))
        self.prev = theano.shared(numpy.array(params['prev']).astype(theano.config.floatX))
        
        idxs_tgt = T.imatrix('y') 
        idxs_char = T.imatrix('c')

        x_tgt = self.emb[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        x_char = self.emb_char[idxs_char].reshape((idxs_char.shape[0], de_char * max_char))
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

        [c, ch, s, s_t],_ = theano.scan(fn=recurrence,
                                        sequences=[x_tgt, x_char],
                                        outputs_info=[self.h0, self.h0_char, self.prev, None], 
					n_steps=x_tgt.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # theano functions
        self.classify = theano.function(inputs=[idxs_tgt, idxs_char], outputs=y_pred)

class GRU_char_bilingual(object):
    
    def __init__(self, params, de, cs, de_char=None, max_char=None):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        #self.emb = theano.shared(numpy.random.normal(0.0, 0.01,\
        #           (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.emb_src =  theano.shared(numpy.array(params['emb_src']).astype(theano.config.floatX))    
        self.emb_tgt =  theano.shared(numpy.array(params['emb_tgt']).astype(theano.config.floatX))    
        normalization_list = [self.emb_src, self.emb_tgt]

        self.emb_char = theano.shared(numpy.array(params['emb_char']).astype(theano.config.floatX)) 
        #Input layer weghts
        self.Wx  = theano.shared(numpy.array(params['Wx']).astype(theano.config.floatX))
        self.Wx_char  = theano.shared(numpy.array(params['Wx_char']).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.array(params['Wh']).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.array(params['bh']).astype(theano.config.floatX))
        self.Wh_char  = theano.shared(numpy.array(params['Wh_char']).astype(theano.config.floatX))
        self.bh_char  = theano.shared(numpy.array(params['bh_char']).astype(theano.config.floatX))
        
        #GRU Gate
        #X(t) weights
        self.Wr = theano.shared(numpy.array(params['Wr']).astype(theano.config.floatX))
        self.Wu = theano.shared(numpy.array(params['Wu']).astype(theano.config.floatX))
        self.Wr_char = theano.shared(numpy.array(params['Wr_char']).astype(theano.config.floatX))
        self.Wu_char = theano.shared(numpy.array(params['Wu_char']).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ur = theano.shared(numpy.array(params['Ur']).astype(theano.config.floatX))
        self.Uu = theano.shared(numpy.array(params['Uu']).astype(theano.config.floatX))
        self.Ur_char = theano.shared(numpy.array(params['Ur_char']).astype(theano.config.floatX))
        self.Uu_char = theano.shared(numpy.array(params['Uu_char']).astype(theano.config.floatX))
        #Bias
        self.br   = theano.shared(numpy.array(params['br']).astype(theano.config.floatX))
        self.bu   = theano.shared(numpy.array(params['bu']).astype(theano.config.floatX))
        self.br_char   = theano.shared(numpy.array(params['br_char']).astype(theano.config.floatX))
        self.bu_char   = theano.shared(numpy.array(params['bu_char']).astype(theano.config.floatX))
    
        #output weights and biases
        self.W   = theano.shared(numpy.array(params['W']).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.array(params['b']).astype(theano.config.floatX))
        self.W_char   = theano.shared(numpy.array(params['W_char']).astype(theano.config.floatX))
        self.b_char   = theano.shared(numpy.array(params['b_char']).astype(theano.config.floatX))
       
        #Recurent memory or h(t-1)
        self.h0  = theano.shared(numpy.array(params['h0']).astype(theano.config.floatX))
        self.h0_char  = theano.shared(numpy.array(params['h0_char']).astype(theano.config.floatX))
        self.prev = theano.shared(numpy.array(params['prev']).astype(theano.config.floatX))
        
        idxs_src = T.imatrix('x') 
        idxs_tgt = T.imatrix('y') 
        idxs_char = T.imatrix('c')

        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        x_char = self.emb_char[idxs_char].reshape((idxs_char.shape[0], de_char * max_char))
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

        [c, ch, s, s_t],_ = theano.scan(fn=recurrence,
                                        sequences=[x_src, x_tgt, x_char],
                                        outputs_info=[self.h0, self.h0_char, self.prev, None], 
					n_steps=x_src.shape[0])
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # theano functions
        self.classify = theano.function(inputs=[idxs_src, idxs_tgt, idxs_char], outputs=y_pred)

