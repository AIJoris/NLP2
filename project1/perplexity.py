import math

def perplexity(e, f, lexicon, eps=1):
    perplexity = 0
    for i in range(len(e)):
        for fw in f[i]:
            sentence_prob = 1
            for ew in e[i]:
                    sentence_prob *= ( eps / (len(e[i])**len(f[i])))*lexicon[ew][fw]
        try:
            perplexity -= math.log(sentence_prob,2)
        # if prob is extremely close to zero, perplexity -= 0. simply skip.
        except:
            pass
    return perplexity
