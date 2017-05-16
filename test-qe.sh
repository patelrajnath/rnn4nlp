#!/bin/bash

python test.py \
--data_test data/qe/test/test.src.lc data/qe/test/test.mt.lc data/qe/test/test.align \
--dictionaries data/qe/train/train.src.lc.json data/qe/train/train.mt.lc.json \
--embeddings data/qe/pretrain/ep_qe.en.vector.txt data/qe/pretrain/ep_qe.de.vector.txt \
--label2index data/qe/train/train.tags.json \
--load_model quest.GRU_adadelta_bilingual_pretrain
#--data_test_y data/qe/test/test.tags \

