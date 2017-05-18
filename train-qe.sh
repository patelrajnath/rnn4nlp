#!/bin/bash

THEANO_FLAGS=optimizer=None python train.py \
--data_train data/qe/train/train.src.lc data/qe/train/train.mt.lc data/qe/train/train.align \
--data_train_y data/qe/train/train.tags \
--data_test data/qe/test/test.src.lc data/qe/test/test.mt.lc data/qe/test/test.align \
--data_test_y data/qe/test/test.tags \
--data_valid data/qe/dev/dev.src.lc data/qe/dev/dev.mt.lc data/qe/dev/dev.align \
--data_valid_y data/qe/dev/dev.tags \
--dictionaries data/qe/train/train.src.lc.json data/qe/train/train.mt.lc.json \
--character2index data/qe/train/train.src.lc.dict_char.json data/qe/train/train.mt.lc.dict_char.json \
--label2index data/qe/train/train.tags.json \
--embeddings data/qe/ep_qe.en.vector.txt data/qe/ep_qe.de.vector.txt \
--use_model GRU \
--saveto \
--use_quest \
--use_char \
#--use_bilingual \
#--use_pretrain # to enable this flag you need to have pretrained word embeddings from word2vec (text)

