from __future__ import division
import numpy
import pdb
import cPickle
import random
import os
import stat
import subprocess, re
from os.path import isfile, join
from os import chmod
import codecs
from sklearn.metrics import f1_score
import numpy as np
from argparse import ArgumentParser
import logging


PREFIX = os.getenv('ATISDATA', '')


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logger = logging.getLogger('wmt_eval_logger')

def read_tag_file(filename):
    with codecs.open(filename) as tagfile:
        tags_by_line = [l.strip().split() for l in tagfile]
    return tags_by_line

def weighted_fmeasure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', pos_label=None)

# scoring results
# print a full score report including statistical significance
def score_wmt_plain(ref_file, hyp_file, n_significance_tests=20):
    ref_tags = read_tag_file(ref_file)
    hyp_tags = read_tag_file(hyp_file)

    assert len(ref_tags) == len(hyp_tags), 'ref file and hyp file must have the same number of tags'
    for ref_line, hyp_line in zip(ref_tags, hyp_tags):
        assert len(ref_line) == len(hyp_line), 'ref line and hyp line must have the same number of tags'

    # flatten out tags
    flat_ref_tags = [t for l in ref_tags for t in l]
    flat_hyp_tags = [t for l in hyp_tags for t in l]

    # EVALUATION
    #logger.info('evaluating your results')

    # look at the actual tag distribution in the reference data
    # TODO: remove the hard coding of the tags here
    bad_count = sum(1 for t in flat_ref_tags if t == u'BAD')
    good_count = sum(1 for t in flat_ref_tags if t == u'OK')

    total = len(flat_ref_tags)
    assert (total == bad_count+good_count), 'tag counts should be correct'
    percent_good = good_count / total
    #logger.info('percent good in test set: {}'.format(percent_good))
    #logger.info('percent bad in test set: {}'.format(1 - percent_good))
    #logger.info('Computing f1 baseline from tag distribution priors')

    random_class_results = []
    random_weighted_results = []
    for i in range(n_significance_tests):
        random_tags = list(np.random.choice([u'OK', u'BAD'], total, [percent_good, 1-percent_good]))
        random_class_f1 = f1_score(flat_ref_tags, random_tags, average=None)
        random_class_results.append(random_class_f1)
        # logger.info('two class f1 random score ({}): {}'.format(i, random_class_f1))
        random_average_f1 = weighted_fmeasure(flat_ref_tags, random_tags)
        random_weighted_results.append(random_average_f1)

    avg_random_class = np.average(random_class_results, axis=0)
    avg_weighted = np.average(random_weighted_results)
    #logger.info('Random Baseline Using the Class priors for \'OK\' and \'BAD\' Tags:')
    #logger.info('two class f1 random average score: {}'.format(avg_random_class))
    #logger.info('weighted f1 random average score: {}'.format(avg_weighted))

    actual_class_f1 = f1_score(flat_ref_tags, flat_hyp_tags, average=None)
    actual_average_f1 = weighted_fmeasure(flat_ref_tags, flat_hyp_tags)
    #logger.info('YOUR RESULTS: ')
    #logger.info('two class f1: {}'.format(actual_class_f1))
    #logger.info('weighted f1: {}'.format(actual_average_f1))
    # END EVALUATION
    return [avg_random_class, avg_weighted, actual_class_f1, actual_average_f1]


def wmt_eval(p, g, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of score_wmt_plain method
    for computing the performance in terms of precision
    recall and f1 score
    '''

    ref = filename+'.ref'
    hyp = filename+'.hyp'
    f_ref = codecs.open(ref , 'w' , 'utf-8')
    f_hyp = codecs.open(hyp , 'w' , 'utf-8')

    for sp, sg in zip(p, g):
	sg1 = re.sub(r'OK_[BIE]', 'OK', ' '.join(sg).strip())
	sp1 = re.sub(r'OK_[BIE]', 'OK', ' '.join(sp).strip())
        f_hyp.write( sp1 + '\n')
        f_ref.write( sg1 +'\n')

    f_hyp.close()
    f_ref.close()  

    return score_wmt_plain(ref, hyp)

if __name__ == '__main__':
    #print get_perf('valid.txt')
    print get_perf('valid.txt')
