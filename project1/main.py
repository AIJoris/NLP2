## Assumptions:
# - Estimate m using uniform distr over all numbers in the range
#   of 1 to max e sentence length
# -

# to do a test:
#perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign naacl.txt

from collections import defaultdict
from load_corpus import load_train, count_words, replace_singletons
import random
import math
import os
from lexicon import init_lexicon
from itertools import permutations, product
from perplexity import perplexity
from viterbi import viterbi, output_naacl
from IBM1_EM import IBM1_EM


## Initialization
# Load parallel corpus
print('Loading data...')
[e,f] = load_train('data', 'train')
# e = e[0:1000]
# f = f[0:1000]

# Find word occurrences for e and f in train data.
# Update singletons to '-LOW-'
count_e, count_f = count_words(e), count_words(f)
e, f = replace_singletons(e, count_e), replace_singletons(f, count_f)

# Initialize lexicon with uniform probabilities
print('Initializing lexicon...')
lexicon = init_lexicon(e, f)

# Expectation Maximization
updated_lexicon = IBM1_EM(e,f,lexicon, nr_it=2)

# use validation set (there is no ground truth for test set)
[e_test,f_test] = load_train('data', 'test')
e_test, f_test = replace_singletons(e_test, count_e), replace_singletons(f_test, count_f)

alignments = viterbi(e_test,f_test,updated_lexicon)

#@TODO: How to treat -NULL- (0th bucket)? maybe do not write it as a position to naacl.txt?
output_naacl(alignments, 'naacl.txt')

# Check file format, calculate scores
os.system("perl data/testing/eval/wa_check_align.pl naacl.txt")
os.system("perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign naacl.txt")




# Determine integer in which m must lie
# M = len(max(e,key=len))
# m = random.choice(range(1,M))
