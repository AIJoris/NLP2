import math

def perplexity(e, f, lexicon, eps=1):
    c_perplexity = 0

    for i in range(len(e)):
        for ew in e[i]:
            s_perplexity = 0
            for fw in f[i]:
                try:
                    s_perplexity = s_perplexity - math.log((eps/(len(f[i])**len(e[i])))*lexicon[ew][fw],2)
                except:
                    # sometimes value is (extremely close to) 0. Log error. Simply skip.
                    pass
        c_perplexity += s_perplexity

    return c_perplexity
