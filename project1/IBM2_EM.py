from collections import defaultdict
import math
import pickle
import os
from load_corpus import load_train, count_words, replace_singletons
from subprocess import PIPE, Popen
import numpy as np
import dill


## Expectation Maximization (EM)
def IBM2_EM(e,f,q,t,max_jump=150,nr_it=10):
    print(' -- EM for IBM 2... ')
    likelihood_values = []
    aer_values = []

    # load test set
    count_e, count_f = count_words(e), count_words(f)
    [e_val,f_val] = load_train('data', 'test')
    e_val, f_val = replace_singletons(e_val, count_e), replace_singletons(f_val, count_f)


    for it in range(nr_it):
        print(' -- Iteration {}... '.format(it+1))
        # Keep track of counts of word co-occurrences
        count_f_e = defaultdict(lambda: defaultdict(float))
        count_e = defaultdict(float)
        # Keep track of counts of jumps
        count_jump = {k:v for (k,v) in zip([i for i in range(-max_jump,max_jump+1)], np.arange(max_jump*2+1)*0)}

        log_likelihood = 0

        print(' -- Expectation... ')
        for (e_sent, f_sent) in zip(e,f):
            sentence_likelihood = 1
            for j, f_w in enumerate(f_sent):
                m = len(e_sent)
                n = len(f_sent)
                # Sum of all alignment link probabilities from current f word to all words in e
                c_t_sum = sum([t[e_w][f_w] for e_w in e_sent])
                # sum all jumps within en sentence length!
                c_j_sum = sum([q[jump_func(i, j, m, n,max_jump)] for i in range(m)])

                #TODO print jump func outputs

                for a_j, e_w in enumerate(e_sent):
                    # Calculate delta
                    index = jump_func(a_j, j, m, n, max_jump)
                    delta = t[e_w][f_w] * q[index]

                    # Update counts
                    count_f_e[e_w][f_w] += delta
                    count_e[e_w] += delta
                    count_jump[index] += delta
                    #TODO normalization factor? Add

                sentence_likelihood *= c_t_sum * c_j_sum
                try:
                    log_likelihood += math.log(sentence_likelihood, 2)
                except ValueError:
                    pass

        print("log likelihood: ",log_likelihood)
        likelihood_values.append(log_likelihood)

        print(' -- Maximization... ')
        for e_w, f_ws in t.items():
            for f_w, _ in f_ws.items():
                try:
                    t[e_w][f_w] = count_f_e[e_w][f_w]/count_e[e_w]
                except ZeroDivisionError:
                    t[e_w][f_w]=0

        total = sum(count_jump.values())
        for jump, p in q.items():
            q[jump] = count_jump[jump]/total

        # Create NAACL file for current run
        output_naacl(best_alignment(e_val,f_val,t,q,max_jump), 'AER/naacl_IBM2_it{}.txt'.format(it+1))
        # Calculate AER values of current lexicon
        aer_values.append(cmdline('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM2_it{}.txt'.format(it+1)))
        os.system('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM2_it{}.txt'.format(it+1))

    print('saving files')
    pickle.dump(likelihood_values, open( "likelihood_IBM2.p", "wb" ) )
    pickle.dump(aer_values, open("AER_IBM2.p", "wb"))
    print('finished saving files')
    return t, q

import random

def jump_func(i, j, m, n, max_jump):
    """
    Alignment of french word j to english word i.
    i = 0, to ,m  (we use m as in Wilker's lecture slides -- length of English sentence)
    j = 1, to ,n  (we use n as in Wilker's lecture slides -- length of French sentence)
    That is: a_j = i
    with e.g. max_jump = 100
    from[-max_jump, max_jump] to [0, 2*max_jump + 1]
    """
    # We normalise j by the lenght of the French sentence and scale the result to the length of the English sentence
    # this gives us a continuous value that is an interpolation of where we j would be in the English sentence
    # if alignments were a linear function of the length ratio
    jump = np.floor(i - (j * m / n))
    # then we collapse all jumps that are too far to the right to the maximum jump value allowed
    if jump > max_jump:  # or we collapse all jumps that are too far to the left to the maximum (negative) jump allowed
        jump = max_jump
    elif jump < -max_jump:
        jump = -max_jump

    return int(jump)


def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]


def best_alignment(e, f, t, q, max_jump):
    print('--Performing Viterbi (IBM 2) --')
    alignments = []
    for i, (e_sent,f_sent) in enumerate(zip(e,f)):
        for j, f_w in enumerate(f_sent):
            best_p = 0
            best_j = 0
            for a_j, e_w in enumerate(e_sent):
                try:
                    if t[e_w][f_w] > best_p:
                        best_p = t[e_w][f_w] * q[jump_func(a_j,j,len(e_sent),len(f_sent),max_jump)]
                        best_j = a_j
                except KeyError:
                    pass
            # +1 because we start counting at 1, i.e. 0 is not allowed.
            # best_j no +1 because of null allignmentself.
            # if best_j is 0, do not save this to allignments. (Because of
            # the way Pearl script expects output)
            if best_j != 0:
                alignments.append((str(i+1).zfill(4), best_j, j+1))
    return alignments

def output_naacl(alignments, fn):
    with open(fn, 'w') as fp:
        fp.write('\n'.join('%s %s %s S' % x for x in alignments))
