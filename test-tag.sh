#!/bin/bash

python test.py \
--data_test data/pos/hi-en.test.txt \
--data_test_y data/pos/hi-en.test.tags \
--dictionaries data/pos/hi-en.train.txt.json \
--label2index data/pos/hi-en.train.tags.json \
--load_model tag.GRU_adadelta
