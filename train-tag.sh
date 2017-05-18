#!/bin/bash

<< comments
python train.py \
--data_train data/pos/icon2016/hi-en.train.txt \
--data_train_y data/pos/icon2016/hi-en.train.tags \
--data_test data/pos/icon2016/hi-en.test.txt \
--data_test_y data/pos/icon2016/hi-en.test.tags \
--data_valid data/pos/icon2016/hi-en.dev.txt \
--data_valid_y data/pos/icon2016/hi-en.dev.tags \
--dictionaries data/pos/icon2016/hi-en.train.txt.json \
--character2index data/pos/icon2016/hi-en.train.txt.dict_char.json \
--label2index data/pos/icon2016/hi-en.train.tags.json \
--use_model GRU \
--use_char \
--saveto \
--use_tag
comments

python train.py \
--data_train data/pos/hi-en.train.txt \
--data_train_y data/pos/hi-en.train.tags \
--data_test data/pos/hi-en.test.txt \
--data_test_y data/pos/hi-en.test.tags \
--data_valid data/pos/hi-en.dev.txt \
--data_valid_y data/pos/hi-en.dev.tags \
--dictionaries data/pos/hi-en.train.txt.json \
--character2index data/poshi-en.train.txt.dict_char.json \
--label2index data/pos/hi-en.train.tags.json \
--use_model GRU \
--saveto \
--use_tag 
