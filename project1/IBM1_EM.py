from collections import defaultdict
import math
from perplexity import perplexity
import pickle
import os
from load_corpus import load_train, count_words, replace_singletons
from viterbi import viterbi, output_naacl


## Expectation Maximization (EM)
def IBM1_EM(e,f,lexicon,nr_it=10):
    perplexity_values = []
    aer_values = []

    # load test set
    count_e, count_f = count_words(e), count_words(f)
    [e_val,f_val] = load_train('data', 'test')
    e_val, f_val = replace_singletons(e_val, count_e), replace_singletons(f_val, count_f)

    print('--Performing EM--')

    for it in range(nr_it):
        # Keep track of counts to be used for the M step
        count_f_e = defaultdict(lambda: defaultdict(int))
        count_e = defaultdict(float)

        # Expectation
        print('Expectation...')
        perplex = 0
        for (e_sent,f_sent) in zip(e,f):
            # Initialize likelihoods
            alignment_likelihood = 1
            sentence_likelihood = 1/(len(e_sent)**len(f_sent))
            for j, f_word in enumerate(f_sent):
                # Sum of all alignment link probabilities from current f word to all words in e
                sum_pi_t = sum([lexicon[e_word][f_word] for e_word in e_sent])
                sentence_likelihood *= sum_pi_t

                for a_j, e_word in enumerate(e_sent):
                    # Single alignment link probability
                    pi_t = lexicon[e_word][f_word]

                    # Ratio between single link and summed link alignment probabilities
                    ratio = pi_t/float(sum_pi_t)
                    alignment_likelihood *= ratio

                    # Update counts
                    count_f_e[e_sent[a_j]][f_sent[j]] += ratio
                    count_e[e_sent[a_j]] += ratio

            if sentence_likelihood > 0:
                # perplex -=math.log(sentence_likelihood,2)
                perplex += math.log(sentence_likelihood,2)
        print('[Iteration {}] perplexity: {}'.format(it+1, round(perplex)))
        perplexity_values.append(perplex)
        # Maximization
        print('Maximization')
        for e_word, f_words in lexicon.items():
            for f_word, prob in f_words.items():
                # print(e_word, f_word, count_f_e[e_word][f_word], float(count_e[e_word]))
                lexicon[e_word][f_word] = count_f_e[e_word][f_word] / float(count_e[e_word])


        # Create NAACL file for current run
        output_naacl(viterbi(e_val,f_val,lexicon), 'AER/naacl_IBM1_it{}.txt'.format(it+1))
        # Calculate AER values of current lexicon
        aer_values.append(cmdline('perl data/testing/eval/wa_eval_align.pl data/testing/answers/test.wa.nonullalign AER/naacl_IBM1_it{}.txt'.format(it+1)))


    pickle.dump(perplexity_values, open( "perplexity_IBM1.p", "wb" ) )
    pickle.dump(aer_values, open("AER_IBM1.p", "wb"))

    return lexicon

from subprocess import PIPE, Popen

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]
