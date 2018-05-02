from collections import defaultdict

def init_lexicon(e, f):
    t_e_f = defaultdict(lambda: defaultdict(int))

    for i in range(len(e)):
        for ew in e[i]:
            for fw in f[i]:
                t_e_f[ew][fw] = 1

    # Normalize to create uniform distribution
    for ew,fwords in t_e_f.items():
        norm=1.0/len(fwords)
        t_e_f[ew] = {k:v*norm for (k,v) in fwords.items()}

    return t_e_f


def init_q(e,f,max_jump,init='uniform'):
    q = defaultdict(lambda: defaultdict(float))

    for e_sent in e:
        # skip 0th word since we are interested in previous alignments
        for i in range(1,len(e_sent)):
            q[len(e_sent)-1][i] += 1
            q[len(e_sent)-1][-i] += 1

    # Normalize to create uniform distribution
    for len_e,pos in q.items():
        norm=1.0/len(pos)
        q[len_e] = {k:norm for (k,v) in pos.items()}

    # q index looks like:  [len(e_sent)-1][jump]
    return q
