import math

def perplexity(e, f, lexicon, eps=1):
    perplexity = 0
    for i, e_sent in enumerate(e):
        for f_word in f[i]:
            translation_likelihood = 1
            for e_word in e_sent:
                translation_likelihood *= ( eps / (len(e_sent)**len(f[i])))*lexicon[e_word][f_word]
        if translation_likelihood > 0:
            perplexity -= math.log(translation_likelihood, 2)

    return perplexity
