# Recurrent Neural Networks for Natural Language Processing (rnn4nlp)
This contains:

(1) RNN based system for word level quality estimation. 

(2) RNN based Parts-of-Speech-Tagger for Code-mixed social media text. 

This include the implementation of varous RNN models including simple Recurrent Neural Network, Long-Short Term Memory,DeepLSTM, and Gated Recurrent Unit. The system is flexible to be used for any word level tagging task like Named Entity Recognition etc.

# Pre-requisites

- python (2.7+)

- Theano (0.8+; http://deeplearning.net/software/theano/)

- numpy (1.11.1+)

- scipy (0.18.0+)

# Quick Start

Quality estimation with toy data:

Create the vocab for training as follows-

```sh

$python utils/build\_dictionary.py data/qe/train/train.src.lc

$python utils/build\_dictionary.py data/qe/train/train.mt.lc

```

And then run the training script-

```sh

$bash train-qe.sh

```
