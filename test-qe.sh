#!/bin/bash

python test.py \
--data_test data/qe/test/test.src.lc data/qe/test/test.mt.lc data/qe/test/test.align \
--data_test_y data/qe/test/test.tags \
--dictionaries data/qe/train/train.src.lc.json data/qe/train/train.mt.lc.json \
--embeddings data/qe/pretrain/ep_qe.en.vector.txt data/qe/pretrain/ep_qe.de.vector.txt \
--character2index data/qe/train/train.src.lc.dict_char.json data/qe/train/train.mt.lc.dict_char.json \
--label2index data/qe/train/train.tags.json \
--load_model quest.LSTM_adadelta_bilingual_pretrain
#--load_model quest.LSTM_pretrain
#--load_model quest.LSTM_adadelta_bilingual
#--load_model quest.LSTM_adadelta
#--load_model quest.LSTM
#--load_model quest.GRU_adadelta_char_bilingual_pretrain
#--load_model quest.GRU_adadelta_bilingual_pretrain
#--load_model quest.GRU_adadelta_pretrain
#--load_model quest.GRU_pretrain
#--load_model quest.GRU_adadelta_char_bilingual
#--load_model quest.GRU_adadelta_bilingual
#--load_model quest.GRU_adadelta_char
#--load_model quest.GRU_adadelta
#--load_model quest.GRU

