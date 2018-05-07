## Assumptions:
# - Estimate m using uniform distr over all numbers in the range
#   of 1 to max e sentence length
# -

# to do a test:
#perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign naacl.txt

from collections    import defaultdict
from load_corpus    import load_train, count_words, replace_singletons
from lexicon        import init_lexicon, init_q
from itertools      import permutations, product
from perplexity     import perplexity
from viterbi        import viterbi, viterbi2, output_naacl
from plots          import make_plots
from IBM1_EM        import IBM1_EM
from IBM2_EM        import IBM2_EM
from IBM1_VB        import IBM1_VB
import random
import math
import os
import pickle
import dill

model = 'IBM1_VB'
nr_it = 10
max_jump = 150
savemodel = True
plots = True
q_init = 'uniform'
# load_lexicon = 'trained_models/save/IBM1_it15_uniform.dill'
load_lexicon = None
# modelfn = 'IBM1'


## Initialization
# Load parallel corpus
print('Loading data...')
# Load train data
[e,f] = load_train('data', 'train')
# Find word occurrences for e and f in train data.
# Update singletons to '-LOW-'
count_e, count_f = count_words(e), count_words(f)
e, f = replace_singletons(e, count_e), replace_singletons(f, count_f)

# Load test data
[e_test,f_test] = load_train('data', 'test')
e_test, f_test = replace_singletons(e_test, count_e), replace_singletons(f_test, count_f)



# Initialize lexicon with uniform probabilities
print('Initializing lexicon...')

if load_lexicon == None:
    lexicon = init_lexicon(e, f)
else:
    lexicon = dill.load(open(load_lexicon, "rb" ))


# Expectation Maximization
if model == 'IBM1':
    print('Training IBM 1')
    trained_lexicon = IBM1_EM(e,f,lexicon,nr_it=nr_it)

    print('calculating final scores...')
    output_naacl(viterbi(e_test, f_test, trained_lexicon), 'AER/naacl_IBM1.txt')
    os.system('perl data/testing/eval/wa_check_align.pl AER/naacl_IBM1.txt')
    os.system('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM1.txt')

    if savemodel: dill.dump(trained_lexicon, open("trained_models/"+model+".dill", "wb" ))

    if plots: make_plots('perplexity_IBM1.p', 'AER_IBM1.p')

elif model == 'IBM1_VB':
    print('Training Bayesian IBM 1')
    e=e[:1000]
    f=f[:1000]
    trained_lexicon = IBM1_VB(e,f,lexicon,nr_it=nr_it)

elif model == 'IBM2':
    print('Initializing q...')
    q = init_q(e,f,max_jump,init=q_init)

    print('Training IBM 2')
    trained_lexicon, trained_q = IBM2_EM(e,f,q,lexicon, max_jump=max_jump, nr_it=nr_it)

    print('finished training')

    if savemodel:
        dill.dump(trained_lexicon, open("trained_models/"+model+"_lexicon.dill", "wb" ))
        dill.dump(trained_q, open("trained_models/"+model+"_q.dill", "wb" ))

    print('calculating final scores...')
    output_naacl(viterbi2(e_test, f_test, trained_lexicon, q, max_jump), 'AER/naacl_IBM2.txt')
    os.system('perl data/testing/eval/wa_check_align.pl AER/naacl_IBM2.txt')
    os.system('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM2.txt')

    if plots: make_plots('perplexity_IBM2.p', 'AER_IBM2.p')




# x=dill.load(open("trained_models/"+model+".dill", "rb" ))
