#!/usr/bin/env python 

import random
import theano
import numpy as np

def add_padding(seqs_x, maxlen):
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)
    x = np.zeros((maxlen, n_samples)).astype('int64')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x[:maxlen]
    return np.transpose(x)

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out


def contextwin_nextWord(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows preceding each word in the sentence
    '''
    
    l = list(l)
    lpadded = win * [-1] + l
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    assert len(out) == len(l)
    return out


if __name__ == '__main__':
	a = [1,2,3,5,6]
	print contextwin_nextWord(a, 4)

