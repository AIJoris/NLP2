## Assumptions:
# - Estimate m using uniform distr over all numbers in the range
#   of 1 to max e sentence length
# -
from collections import defaultdict
from load_corpus import load_train
import random
from lexicon import init_lexicon
from itertools import permutations, product

## Initialization
# Load parallel corpus
print('Loading data...')
[e,f] = load_train('data')
e = e[0:1000]
f = f[0:1000]

# Initialize lexicon with uniform probabilities
print('Initializing lexicon...')
lexicon = init_lexicon(e, f)

# Determine integer in which m must lie
M = len(max(e,key=len))
m = random.choice(range(1,M))

## Expectation Maximization (EM)
print('Performing EM')
#TODO add perplexity comparison
while True:
    # Keep track of counts to be used for the M step
    count_f_e = defaultdict(lambda: defaultdict(int)) #[e_word][f_word]
    count_e = defaultdict(int) #TODO should be count_f?

    # Expectation
    print('Expectation...')
    for (e_sent,f_sent) in zip(e,f):
        for j, f_word in enumerate(f_sent):
            sum_pi_t = sum([lexicon[e_word][f_word] for e_word in e_sent])
            for a_j, e_word in enumerate(e_sent):
                pi_t = lexicon[e_word][f_word]
                ratio = pi_t/float(sum_pi_t)

                # Update counts
                count_f_e[e_sent[a_j]][f_sent[j]] += ratio
                count_e[e_sent[a_j]] += ratio

    # Maximization
    print('Maximization')
    for e_word,f_words in lexicon.items():
        for f_word, prob in f_words.items():
            lexicon[e_word][f_word] = count_f_e[e_word][f_word]/float(count_e[e_word])
