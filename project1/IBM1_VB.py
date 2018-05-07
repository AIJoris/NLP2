from collections import defaultdict
import math
from perplexity import perplexity
import pickle
from scipy.special import digamma

## Expectation Maximization (EM)
def IBM1_VB(e,f,Lambda,nr_it=10,alpha=0.01):


    # TODO  log likelihood

    print('--Performing EM--')
    for it in range(nr_it):
        print('Expectation...')

        c_e_f = defaultdict(lambda: defaultdict(int))

        for (e_sent,f_sent) in zip(e,f):
            for f_w in f_sent:
                for e_w in e_sent:
                    # This is theta: note that it is a function of Lambda.
                    # Use theta to update the counts
                    theta = math.exp(digamma(Lambda[e_w][f_w]) - digamma(sum(Lambda[e_w].values())))
                    c_e_f[e_w][f_w] += theta
        print('Maximization')
        for e_w, f_words in theta:
            for f_w, p in f_words.items():
                Lambda[e_w][f_w] = alpha + c_e_f[e_w][f_w]

    return Lamba
