from collections import defaultdict

def init_t_e_f(e, f):
    t_e_f = defaultdict(lambda: defaultdict(int))

    for i in range(len(e)):
        for ew in e[i]:
            t_e_f[ew]['NULL'] = 1
            for fw in f[i]:
                t_e_f[ew][fw] = 1

    # Normalize to create uniform distribution
    for ew,fwords in t_e_f.items():
        norm=1.0/len(fwords)
        t_e_f[ew] = {k:v*norm for (k,v) in fwords.items()}

    return t_e_f
