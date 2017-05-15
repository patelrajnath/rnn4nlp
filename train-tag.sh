#!/bin/bash

python train.py \
--data_train data/pos/hi-en.train.txt \
--data_train_y data/pos/hi-en.train.tags \
--data_test data/pos/hi-en.test.txt \
--data_test_y data/pos/hi-en.test.tags \
--data_valid data/pos/hi-en.dev.txt \
--data_valid_y data/pos/hi-en.dev.tags \
--dictionaries data/pos/hi-en.train.txt.json \
--saveto \
--use_tag 

