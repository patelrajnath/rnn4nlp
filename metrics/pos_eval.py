from __future__ import division
import codecs
from sklearn.metrics import f1_score
import logging, sys


def read_tag_file(filename):
    with codecs.open(filename) as tagfile:
        tags_by_line = [l.strip().split() for l in tagfile]
    return tags_by_line

def weighted_fmeasure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', pos_label=None)

def score_icon_plain(ref_file, hyp_file, n_significance_tests=20):

    ref_tags = read_tag_file(ref_file)
    hyp_tags = read_tag_file(hyp_file)

    assert len(ref_tags) == len(hyp_tags), 'ref file and hyp file must have the same number of tags'
    for ref_line, hyp_line in zip(ref_tags, hyp_tags):
        assert len(ref_line) == len(hyp_line), 'ref line and hyp line must have the same number of tags'

    # flatten out tags
    flat_ref_tags = [t for l in ref_tags for t in l]
    flat_hyp_tags = [t for l in hyp_tags for t in l]

    actual_class_f1 = f1_score(flat_ref_tags, flat_hyp_tags, average=None)
    actual_average_f1 = weighted_fmeasure(flat_ref_tags, flat_hyp_tags)

    # END EVALUATION
    return [actual_class_f1, actual_average_f1]

def icon_eval(p, g, filename):
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
	sg1 = ' '.join(sg).strip()
	sp1 = ' '.join(sp).strip()
        f_hyp.write( sp1 + '\n')
        f_ref.write( sg1 +'\n')

    f_hyp.close()
    f_ref.close()  

    return score_icon_plain(ref, hyp)

if __name__ == '__main__':
	print icon_eval(sys.argv[1], sys.argv[2])
