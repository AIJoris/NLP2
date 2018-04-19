## Assumptions:
# - Estimate m using uniform distr over all numbers in the range
#   of 1 to max e sentence length
# -
from collections import defaultdict
from load_corpus import load_train
import random
import math
from lexicon import init_lexicon
from itertools import permutations, product
from perplexity import perplexity

from IBM1_EM import IBM1_EM

## Initialization
# Load parallel corpus
print('Loading data...')
[e,f] = load_train('data')
e = e[0:1000]
f = f[0:1000]

# Initialize lexicon with uniform probabilities
print('Initializing lexicon...')
lexicon = init_lexicon(e, f)

updated_lexicon = IBM1_EM(e,f,lexicon)

# Determine integer in which m must lie
M = len(max(e,key=len))
m = random.choice(range(1,M))
