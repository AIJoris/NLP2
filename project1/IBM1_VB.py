from collections import defaultdict
import math
from perplexity import perplexity
import pickle
from scipy.special import digamma
import os
from load_corpus import load_train, count_words, replace_singletons
from viterbi import viterbi, output_naacl

## Expectation Maximization (EM)
def IBM1_VB(e,f,Lambda,nr_it=10,alpha=0.01):

    aer_values = []

    # load test set
    [e_val,f_val] = load_train('data', 'test')
    e_val, f_val = replace_singletons(e_val, count_words(e)), replace_singletons(f_val, count_words(f))

    print('--Performing EM--')
    for it in range(nr_it):
        print('Expectation...')

        c_e_f = defaultdict(lambda: defaultdict(float))

        for (e_sent,f_sent) in zip(e,f):
            for f_w in f_sent:
                for e_w in e_sent:
                    # This is theta: note that it is a function of Lambda.
                    # Use theta to update the counts
                    theta = math.exp(digamma(Lambda[e_w][f_w]) - digamma(sum(Lambda[e_w].values())))
                    c_e_f[e_w][f_w] += theta
        print('Maximization')

        for e_w, f_words in Lambda.items():
            for f_w, p in f_words.items():
                Lambda[e_w][f_w] = alpha + c_e_f[e_w][f_w]


        # Create NAACL file for current run
        output_naacl(viterbi(e_val,f_val,Lambda), 'AER/naacl_IBM1VB_it{}.txt'.format(it+1))
        os.system('perl data/testing/eval/wa_check_align.pl AER/naacl_IBM1VB_it{}.txt'.format(it+1))

        os.system('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM1VB_it{}.txt'.format(it+1))
    return Lambda
