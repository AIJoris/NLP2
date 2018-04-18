import math

def perplexity(e, f, lexicon, eps=1):
    perplexity = 0
    for i in range(len(e)):
        for fw in f[i]:
            sentence_prob = 1
            for ew in e[i]:
                    sentence_prob *= ( eps / (len(e[i])**len(f[i])))*lexicon[ew][fw]
        perplexity -= math.log(sentence_prob,2)
    return perplexity
