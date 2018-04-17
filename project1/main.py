## Assumptions:
# - Estimate m using uniform distr over all numbers in the range
#   of 1 to max e sentence length
# -

from load_corpus import load_train
import random
from lexicon import init_t_e_f

## Initialization
# Load parallel corpus
[e,f] = load_train('data')

# Initialize lexicon with uniform probabilities
lexicon = init_t_e_f(english, french)

# Determine integer in which m must lie
M = len(max(e,key=len))
e.sort(key=len, reverse=True)

## EM
# Expectation
E_likelihood = 0
for (e_sent,f_sent) in zip(e,f):
    # For every sentence, compute P(a|e,f)
    for f_word in f_sent:
        sum_pi_t = 0
        for e_word in e_sent:
            sum_pi_t += lexicon[e_word][f_word]
        pi_t = lexicon[e_word][f_word]

    # Estimate translation sentence length
    # m = random.choice(range(1,M))

#     # Compute likelihood for all possible alignments
#     print(e_sent,m)
#     break
