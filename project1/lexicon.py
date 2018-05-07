from collections import defaultdict

def init_lexicon(e, f,init="random"):
    t_e_f = defaultdict(lambda: defaultdict(float))

    if init == "random":
        for i in range(len(e)):
            for ew in e[i]:
                for fw in f[i]:
                    t_e_f[ew][fw] = random.random()

    else:
        for i in range(len(e)):
            for ew in e[i]:
                for fw in f[i]:
                    t_e_f[ew][fw] = 1

        # Normalize to create uniform distribution
        for ew,fwords in t_e_f.items():
            norm=1.0/len(fwords)
            t_e_f[ew] = {k:v*norm for (k,v) in fwords.items()}

    return t_e_f

import random

def init_q(e,f,max_jump,init='uniform'):
    if init == 'uniform':
        uniform = 1. / (2 * max_jump)
        q = {k:v for (k,v) in zip([i for i in range(-max_jump,max_jump+1)], [uniform for i in range(-max_jump,max_jump+1)])}

    elif init == 'random':
        q = {k:v for (k,v) in zip([i for i in range(-max_jump,max_jump+1)], [random.random() for i in range(-max_jump,max_jump+1)])}
        q = {k:v/sum(q.values()) for (k,v) in q.items()}
    return q
