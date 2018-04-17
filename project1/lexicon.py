from collections import defaultdict

def create_lexicon(e, f):
    count_f = defaultdict(int)
    count_e_f = defaultdict(lambda: defaultdict(int))

    for i in range(len(e)):
        # create counts for French words per sentence
        for f_w in f[i]:
            count_f[f_w] += 1

        # find English French counts
        for e_w in e[i]:
            for f_w in f[i]:
                count_e_f[e_w][f_w] += 1

    # find values for pi_t
    pi_t = defaultdict(lambda: defaultdict(float))

    for e_w,f_words in count_e_f.items():
        for f_w,f_w_count in f_words.items():
            pi_t[e_w][f_w] = f_w_count / count_f[f_w]

    return pi_t
