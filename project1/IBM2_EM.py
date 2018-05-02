from collections import defaultdict
import math
from perplexity import perplexity
import pickle
import os
from load_corpus import load_train, count_words, replace_singletons
from viterbi import viterbi, viterbi_IBM2, output_naacl
from subprocess import PIPE, Popen
import numpy as np


## Expectation Maximization (EM)
def IBM2_EM(e,f,lexicon,q,max_jump,nr_it=10):
    perplexity_values = []
    aer_values = []

    # load test set
    count_e, count_f = count_words(e), count_words(f)
    [e_val,f_val] = load_train('data', 'test')
    e_val, f_val = replace_singletons(e_val, count_e), replace_singletons(f_val, count_f)

    print('--Performing EM--')

    for it in range(nr_it):
        # Keep track of counts of word co-occurrences to be used for the M step
        count_f_e = defaultdict(lambda: defaultdict(int))
        count_e = defaultdict(float)

        # Keep track of alignment counts
        count_M_aj_ajprev = defaultdict(lambda: defaultdict(float))
        count_M_ajprev = defaultdict(float)

        # Expectation
        print('Expectation...')
        perplex = 0
        sen_i=1
        for (e_sent,f_sent) in zip(e,f):
            # Initialize likelihoods
            lexicon_likelihood = 1
            alignment_likelihood = 1
            sentence_likelihood = 1/(len(e_sent)**len(f_sent))
            # sentence_likelihood = 1
            if sen_i%100 == 0: print('sentence ',sen_i,'/',len(e),', it=',it+1)
            sen_i+=1
            for j, f_word in enumerate(f_sent):
                # Sum of all alignment link probabilities from current f word to all words in e
                sum_pi_t = sum([lexicon[e_word][f_word] for e_word in e_sent])


                for a_j, e_word in enumerate(e_sent):
                    ###############################
                    ##### LEXICON PROBABILITY #####
                    ###############################
                    # Single world probability
                    pi_t = lexicon[e_word][f_word]
                    # Update lexicon_likelihood
                    ratio_lex = pi_t/float(sum_pi_t)
                    lexicon_likelihood *= ratio_lex

                    # Update counts
                    count_f_e[e_sent[a_j]][f_sent[j]] += ratio_lex
                    count_e[e_sent[a_j]] += ratio_lex

                    ###############################
                    #### ALIGNMENT PROBABILITY ####
                    ###############################
                    for a_j_prev in range(1,len(e_sent)):
                        if a_j != a_j_prev and abs(a_j-a_j_prev) <= max_jump:

                            #TODO: how to define sum q ajprev? like this or like above?
                            # This follows eq. 2.6 from https://www.cs.sfu.ca/~anoop/students/anahita_mansouri/anahita-depth-report.pdf
                            sum_q_ajprev = 0
                            for i in range(len(e_sent)):
                                if i-a_j_prev != 0:
                                    # print(i-a_j_prev)
                                    sum_q_ajprev += q[len(e_sent)-1][i-a_j_prev]
                            # print('s',sum_q_ajprev)

                            # Single alignment probability
                            q_aj_ajprev = q[len(e_sent)-1][a_j-a_j_prev]
                            # Update alignment likelihood
                            ratio_q = q_aj_ajprev/float(sum_q_ajprev)
                            alignment_likelihood *= ratio_q
                            # Update counts
                            count_M_aj_ajprev[len(e_sent)-1][a_j-a_j_prev] += ratio_q
                            count_M_ajprev[len(e_sent)-1] += ratio_q



                sentence_likelihood *= (sum_pi_t * sum_q_ajprev)
                # print(sum_pi_t, sum_q_ajprev)
                # print('sl',sentence_likelihood)




            if sentence_likelihood > 0:
                # perplex -=math.log(sentence_likelihood,2)
                perplex += math.log(sentence_likelihood,2)
        print('[Iteration {}] perplexity: {}'.format(it+1, round(perplex)))
        perplexity_values.append(perplex)


        # Maximization
        print('Maximization')
        for e_word, f_words in lexicon.items():
            for f_word, prob in f_words.items():
                lexicon[e_word][f_word] = count_f_e[e_word][f_word] / float(count_e[e_word])

        for e_len, positions in q.items():
            for pos, p in positions.items():
                q[e_len][pos] = count_M_aj_ajprev[e_len][pos]/float(count_M_ajprev[e_len])
                #Results seem to make sense: diagonal alignments higher probability



        #TODO: alignment precisely opposite? jump 1 lower prob than jump -1, jump 2 lower prob than jump -2, ...
        # viterbi_IBM2(e,f, lexicon, q)
        # Create NAACL file for current run
        # output_naacl(viterbi_IBM2(e_val,f_val,lexicon, q), 'AER/naacl_IBM2_it{}.txt'.format(it+1))


        # Calculate AER values of current lexicon
        # aer_values.append(cmdline('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM2_it{}.txt'.format(it+1)))

    pickle.dump(perplexity_values, open( "perplexity_IBM2.p", "wb" ) )
    pickle.dump(aer_values, open( "AER_IBM2.p", "wb" ) )
    return lexicon, q


def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]
