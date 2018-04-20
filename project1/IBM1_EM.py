from collections import defaultdict
import math
from perplexity import perplexity
import pickle

## Expectation Maximization (EM)
def IBM1_EM(e,f,lexicon,nr_it=10):
    print('--Performing EM--')

    for it in range(nr_it):
        # Keep track of counts to be used for the M step
        count_f_e = defaultdict(lambda: defaultdict(int))
        count_e = defaultdict(int)

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

            # TODO should we skip big/unlikely sentences in this way?
            if sentence_likelihood > 0:
                perplex -=math.log(sentence_likelihood,2)
        print('[Iteration 0] perplexity: {}'.format(round(perplex)))
        # Maximization
        print('Maximization')
        for e_word, f_words in lexicon.items():
            for f_word, prob in f_words.items():
                lexicon[e_word][f_word] = count_f_e[e_word][f_word] / float(count_e[e_word])

    return lexicon
