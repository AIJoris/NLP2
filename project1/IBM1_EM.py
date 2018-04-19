from collections import defaultdict
import math
from perplexity import perplexity
import pickle

## Expectation Maximization (EM)
def IBM1_EM(e,f,lexicon):
    print('Performing EM')
    perplex = perplexity(e, f, lexicon)
    updated_perplex = 0

    while True:
        perplex = updated_perplex

        # Keep track of counts to be used for the M step
        count_f_e = defaultdict(lambda: defaultdict(int))
        count_e = defaultdict(int)
        count_f = defaultdict(int)

        # Expectation
        print('Expectation...')
        local_perplex = 0
        for (e_sent,f_sent) in zip(e,f):
            # Initialize likelihoods
            alignment_likelihood = 1
            sentence_likelihood = 1/float(len(e_sent)**len(f_sent))
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
                    count_f[f_word] += ratio
            if sentence_likelihood > 0:
                local_perplex -=math.log(sentence_likelihood,2)

        # Maximization
        print('Maximization')
        for e_word, f_words in lexicon.items():
            for f_word, prob in f_words.items():
                lexicon[e_word][f_word] = count_f_e[e_word][f_word] / float(count_e[e_word])
                # lexicon[e_word][f_word] = count_f_e[e_word][f_word] / float(count_f[f_word])

        updated_perplex = perplexity(e,f,lexicon)
        print('local',local_perplex)
        print('updated',updated_perplex)

    return lexicon
