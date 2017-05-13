import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict
from is13.utils.tools import numpy_floatX

class model_with_pretraining(object):
    
    def __init__(self, nh, nc, de, cs, emb_tgt, emb_src, nh2=100):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        #self.emb = theano.shared(0.2 * numpy.random.uniform(-0.01, 0.01,\
        #           (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        
        self.emb_src = theano.shared(numpy.asarray(emb_src).astype(theano.config.floatX)) 
        self.emb_tgt = theano.shared(numpy.asarray(emb_tgt).astype(theano.config.floatX)) 

        normalization_list = [self.emb_src, self.emb_tgt]
        
        #Input layer weghts
        self.Wx  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        
        #Reccurant weight or g(t) in note
        self.Wh  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        #LSTM Gate
        #X(t) weights
        self.Wi = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wo = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        self.Wf = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (2 * de * cs, nh)).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ui = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uo = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        self.Uf = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh, nh))[0].astype(theano.config.floatX))
        #Bias
        self.bi   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bo   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bf   = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        
        
        #Weights for hidden layer-2
        #Input layer weghts
        self.Wx2  = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nh2)).astype(theano.config.floatX))
        #Reccurant weight or g(t) in note
        self.Wh2  = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                    nh2, nh2))[0].astype(theano.config.floatX))
        self.bh2  = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        #LSTM Gate
        #X(t) weights
        self.Wi2 = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nh2)).astype(theano.config.floatX))
        self.Wo2 = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nh2)).astype(theano.config.floatX))
        self.Wf2 = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh, nh2)).astype(theano.config.floatX))
        #H(t-1) weights
        self.Ui2 = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh2, nh2))[0].astype(theano.config.floatX))
        self.Uo2 = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh2, nh2))[0].astype(theano.config.floatX))
        self.Uf2 = theano.shared(numpy.linalg.svd(numpy.random.randn(\
                   nh2, nh2))[0].astype(theano.config.floatX))
        #Bias
        self.bi2   = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        self.bo2   = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        self.bf2   = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))

        #output weights and biases
        self.W   = theano.shared(numpy.random.normal(0.0, 0.01,\
                   (nh+nh2, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX)) 
        
        #Recurent memory
        self.h1  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.c1  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
  
        self.h2  = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        self.c2  = theano.shared(numpy.zeros(nh2, dtype=theano.config.floatX))
        
        # bundle
        self.params = [ self.emb_src, self.emb_tgt, self.Wx, self.Wh, self.Wi, self.Wo, self.Wf, self.Wx2, self.Wh2, self.Wi2, self.Wo2, self.Wf2, self.W,\
                       self.Ui, self.Uo, self.Uf,  self.Ui2, self.Uo2, self.Uf2,\
                       self.bh, self.bi, self.bo, self.bf, self.bh2, self.bi2, self.bo2, self.bf2, self.b,\
                       self.h1, self.c1, self.h2, self.c2]

        
        self.names  = ['emb_src', 'emb_tgt', 'Wx', 'Wh', 'Wi', 'Wo', 'Wf','Wx2', 'Wh2', 'Wi2', 'Wo2', 'Wf2', 'W', \
                       'Ui', 'Uo', 'Uf',  'Ui2', 'Uo2', 'Uf2',\
                       'bh', 'bi', 'bo', 'bf', 'bh2', 'bi2', 'bo2', 'bf2',\
                       'b', 'h1', 'c1', 'h2', 'c2']
        
        idxs_src =  T.imatrix()
        x_src = self.emb_src[idxs_src].reshape((idxs_src.shape[0], de*cs))
        idxs_tgt = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x_tgt = self.emb_tgt[idxs_tgt].reshape((idxs_tgt.shape[0], de*cs))
        
        y    = T.ivector('y') # label
        
        def recurrence(x_src, x_tgt, h_tm1, c_, h_tm2, c2_):
            
            x_t = T.concatenate([x_src, x_tgt])
            
            g_i = T.nnet.sigmoid(T.dot(x_t, self.Wi) + T.dot(h_tm1, self.Ui) + self.bi)
            g_o = T.nnet.sigmoid(T.dot(x_t, self.Wo) + T.dot(h_tm1, self.Uo) + self.bo)
            g_f = T.nnet.sigmoid(T.dot(x_t, self.Wf) + T.dot(h_tm1, self.Uf) + self.bf)
            
            g_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            
            c1 = g_f*c_ + g_i*g_t

            h1 = g_o*T.tanh(c1)
            
            g_i2 = T.nnet.sigmoid(T.dot(h1, self.Wi2) + T.dot(h_tm2, self.Ui2) + self.bi2)
            g_o2 = T.nnet.sigmoid(T.dot(h1, self.Wo2) + T.dot(h_tm2, self.Uo2) + self.bo2)
            g_f2 = T.nnet.sigmoid(T.dot(h1, self.Wf2) + T.dot(h_tm2, self.Uf2) + self.bf2)
            
            g_t2 = T.tanh(T.dot(h1, self.Wx2) + T.dot(h_tm2, self.Wh2) + self.bh2)
            
            c2 = g_f2*c2_ + g_i2*g_t2

            h2 = g_o2*T.tanh(c2)
            
            h3 = T.concatenate([h1 , h2])
            s_t = T.nnet.softmax(T.dot(h3, self.W) + self.b)
            return [h1, c1, h2, c2, s_t]

        [h1, c1, h2, c2, s_t], _ = theano.scan(fn=recurrence, \
            sequences=[x_src, x_tgt], outputs_info=[self.h1, self.c1, self.h2, self.c2, None], \
            n_steps=x_src.shape[0], truncate_gradient=7)

        #p_y_given_x_lastword = s[-1,0,:]
        
        p_y_given_x_sentence = s_t[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -(T.log(s_t[:,0,:])[T.arange(y.shape[0]), y]).sum()
        
        grads = T.grad( nll, self.params)
        
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
        
        self.train_grad_shared = theano.function(inputs=[idxs_src, idxs_tgt, y, lr], outputs=nll, updates=zgup + rg2up,
                                    on_unused_input='ignore', name='adadelta_train_grad_shared', allow_input_downcast=True)
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
