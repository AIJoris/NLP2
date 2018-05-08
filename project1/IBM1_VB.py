from collections import defaultdict
import math
import pickle
from scipy.special import digamma
import os
from load_corpus import load_train, count_words, replace_singletons
from viterbi import viterbi, output_naacl
import pickle
import dill
from lexicon import init_lexicon

## Expectation Maximization (EM)
def IBM1_VB(e,f,Lambda,nr_it=10,alpha=0.01):

    aer_values = []
    elbo_values = []

    # load test set
    count_e = count_words(e)
    count_f = count_words(f)
    theta = init_lexicon(e,f,init="uniform")

    [e_val,f_val] = load_train('data', 'test')
    e_val, f_val = replace_singletons(e_val, count_e), replace_singletons(f_val, count_f)

    print('--Performing EM--')
    for it in range(nr_it):
        print('Expectation...')
        count_f_e = defaultdict(lambda: defaultdict(float))

        for sentnr,(e_sent,f_sent) in enumerate(zip(e,f)):
            if sentnr % 10000 == 0:
                print('#sent:',sentnr)
            for f_w in f_sent:
                sum_pi_t = sum([theta[e_word][f_w] for e_word in e_sent])
                for e_w in e_sent:
                    pi_t = theta[e_w][f_w]

                    # Update counts
                    count_f_e[e_w][f_w] += pi_t/sum_pi_t

        print('Maximization')
        for e_w, f_words in Lambda.items():
            X = digamma(sum(Lambda[e_w].values()))
            for f_w, p in f_words.items():
                Lambda[e_w][f_w] = alpha + count_f_e[e_w][f_w]
                theta[e_w][f_w] = math.exp(digamma(Lambda[e_w][f_w]) - X)



        #TODO: calculate ELBO
        elbo = calculate_elbo(e,f,count_e,count_f,theta,alpha)
        print('ELBO:',elbo)
        elbo_values.append(elbo)

        # Create NAACL file for current run
        output_naacl(viterbi(e_val,f_val,theta), 'AER/naacl_IBM1VB_it{}.txt'.format(it+1))
        aer_values.append(cmdline('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM1VB_it{}.txt'.format(it+1)))
        os.system('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM1VB_it{}.txt'.format(it+1))

    # pickle.dump(elbo_values, open( "ELBO_IBM_VI.p", "wb" ) )
    pickle.dump(aer_values, open("AER_IBM_VI.p", "wb"))
    return theta, elbo_values

from scipy.special import gammaln

def calculate_elbo(e,f,count_e, count_f,theta,alpha):
    print('Calculating ELBO ...')
    # Count_e, count_f are dicts with vocabulary (counts), can be used for second part

    # see https://uva-slpl.github.io/nlp2/projects/2018/05/03/ELBO.html
    # First part
    elbo_p1 = 0
    for (e_sent,f_sent) in zip(e,f):
        uniform_q = 1/(len(e_sent)**len(f_sent))
        for f_w in f_sent:
            elbo_p1 += uniform_q * sum([theta[e_w][f_w] for e_w in e_sent])

    # Second part
    elbo_p2 = 0
    sum_f_a = len(count_f)*alpha

    # Loop over french vocabulary
    for e_w in count_e.keys():
        try:
            sum_f_VF = sum([theta[e_w][f_word] for f_word in count_f.keys()])
        except KeyError:
            continue
        for f_w in count_f.keys():

        # Loop over english vocabulary
            elbo_p2 += math.exp(digamma(theta[e_w][f_w]) - digamma(sum_f_VF)) \
             * (alpha-theta[e_w][f_w]) \
             + gammaln(sum_f_a) \
             - gammaln(sum_f_VF)
    return elbo_p1 + elbo_p2

from subprocess import PIPE, Popen

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]
