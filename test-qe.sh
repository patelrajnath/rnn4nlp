#!/bin/bash

python test.py \
--data_test data/qe/test/test.src.lc data/qe/test/test.mt.lc data/qe/test/test.align \
--data_test_y data/qe/test/test.tags \
--dictionaries data/qe/train/train.src.lc.json data/qe/train/train.mt.lc.json \
--embeddings data/qe/pretrain/ep_qe.en.vector.txt data/qe/pretrain/ep_qe.de.vector.txt \
--label2index data/qe/train/train.tags.json \
--load_model quest.GRU_adadelta

