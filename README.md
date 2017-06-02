## Recurrent Neural Networks for Natural Language Processing (rnn4nlp)
This repository contains:

(1) RNN based system for word level quality estimation. 

(2) RNN based Part-of-Speech tagger for code-mixed social media text. 

This includes the implementation of various RNN models including simple Recurrent Neural Network, Long-Short Term Memory (LSTM), DeepLSTM, and Gated Recurrent Units (GRU) aka Gated Hidden Units (GHU). The system is flexible to be used for any word level NLP tagging task like Named Entity Recognition etc.

## Pre-requisites

- python (2.7+)

- Theano (0.8+; http://deeplearning.net/software/theano/)

- numpy (1.11.1+)

- python-sklearn (0.17+)

## Quick Start

#### Quality estimation with toy data:

Create the vocab for training-

```sh

WORD INPUT
$python utils/build_dictionary.py data/qe/train/train.src.lc 0
$python utils/build_dictionary.py data/qe/train/train.mt.lc 0

CHARACTER INPUT (--use_char switch)
$python utils/build_char_dictionary.py data/qe/train/train.src.lc
$python utils/build_char_dictionary.py data/qe/train/train.mt.lc


LABELS
$python utils/build_dictionary.py data/qe/train/train.tags 1

```

Note: --use_char switch is available only with GRU model

And then run the training script-

```sh

$bash train-qe.sh

```
Testing with new test-set-

```sh

$bash test-qe.sh

```
Note: You can specifiy any text for testing but dictionaries and label2index should be the same as used at training time

#### Part-of-Speech tagging with toy data:

Create vocab for training-

```sh

WORD INPUT
$python utils/build_dictionary.py data/pos/hi-en.train.txt 0

CHARACTER INPUT (-use_char switch)
$python utils/build_char_dictionary.py data/pos/hi-en.train.txt

LABELS
$python utils/build_dictionary.py data/pos/hi-en.train.tags 1

```

#

And then run the training script-

```sh 

$bash train-tag.sh

```
Testing with new test-set-

```sh

$bash test-tag.sh

```
Note: You can specifiy any text for testing but dictionaries and label2index should be the same as used at training time

#### Detailed Description

For detailed description visit the wiki page-
https://github.com/patelrajnath/rnn4nlp/wiki

## Publications:

If you use this project, please cite the following papers:

> @InProceedings{patel-m:2016:WMT,
>  author    = {Patel, Raj Nath  and  M, Sasikumar},
>  title     = {Translation Quality Estimation using Recurrent Neural Network},
>  booktitle = {Proceedings of the First Conference on Machine Translation},
>  month     = {August},
>  year      = {2016},
>  address   = {Berlin, Germany},
>  publisher = {Association for Computational Linguistics},
>  pages     = {819--824},
>  url       = {http://www.statmt.org/wmt16/pdf/W16-2389.pdf } }

 
> @article{patel2016recurrent,
>  title={Recurrent Neural Network based Part-of-Speech Tagger for Code-Mixed Social Media Text},
>  author={Patel, Raj Nath and Pimpale, Prakash B and Sasikumat, M},
>  journal={arXiv preprint arXiv:1611.04989},
>  year={2016}
>  url = {https://arxiv.org/pdf/1611.04989.pdf } }


## Author 

Raj Nath Patel (patelrajnath@gmail.com)

Linkedin: https://www.linkedin.com/in/raj-nath-patel-2262b024/

## Version

0.1

## LICENSE

Copyright Raj Nath Patel 2017 - present

rnn4nlp is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

You should have received a copy of the GNU General Public License along with Indic NLP Library. If not, see http://www.gnu.org/licenses/.

