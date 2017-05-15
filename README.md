# Recurrent Neural Networks for Natural Language Processing (rnn4nlp)
This contains:

(1) RNN based system for word level quality estimation. 

(2) RNN based Parts-of-Speech-Tagger for Code-mixed social media text. 

This includes the implementation of various RNN models including simple Recurrent Neural Network, Long-Short Term Memory (LSTM), DeepLSTM, and Gated Recurrent Units (GRU) aka Gated Hidden Units (GHU). The system is flexible to be used for any word level NLP tagging task like Named Entity Recognition etc.

# Pre-requisites

- python (2.7+)

- Theano (0.8+; http://deeplearning.net/software/theano/)

- numpy (1.11.1+)

- scipy (0.18.0+)

# Quick Start

### Quality estimation with toy data:

Create the vocab for training-

```sh

$python utils/build_dictionary.py data/qe/train/train.src.lc
$python utils/build_dictionary.py data/qe/train/train.mt.lc

```

And then run the training script-

```sh

$bash train-qe.sh

```

### Parts-of-Speech tagging with toy data:

Create vocab for training-

```sh

$python utils/build_dictionary.py data/pos/hi-en.train.txt

```

And then run the training script-

```sh 

$bash train-tag.sh

```

## Author 

Raj Nath Patel (patelrajnath@gmail.com)

Linkedin: https://www.linkedin.com/in/raj-nath-patel-2262b024/

## Version

0.1

## LICENSE

Copyright Raj Nath Patel 2017 - present

rnn4nlp is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

You should have received a copy of the GNU General Public License along with Indic NLP Library. If not, see http://www.gnu.org/licenses/.

