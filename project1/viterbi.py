from load_corpus import load_train
from lexicon import init_lexicon
import numpy as np
from collections import defaultdict
import math

def viterbi(e, f, lexicon):
    print('--Performing Viterbi--')
    alignments = []
    for i, (e_sent,f_sent) in enumerate(zip(e,f)):
        for j, f_w in enumerate(f_sent):
            best_p = 0
            best_j = 0
            for a_j, e_w in enumerate(e_sent):
                try:
                    if lexicon[e_w][f_w] > best_p:
                        best_p = lexicon[e_w][f_w]
                        best_j = a_j
                except KeyError:
                    pass
            # +1 because we start counting at 1, i.e. 0 is not allowed.
            # best_j no +1 because of null allignmentself.
            # if best_j is 0, do not save this to allignments. (Because of
            # the way Pearl script expects output)
            if best_j != 0:
                alignments.append((str(i+1).zfill(4), best_j, j+1))
    return alignments

def viterbi_IBM2(e, f, pi, q):
    alignments = []

    print('--- Performing Viterbi ---   ')
    for sen_count, (e_sent, f_sent) in enumerate(zip(e,f)):
        # Initialize V
        V = np.zeros((len(e_sent), len(f_sent)))
        trace = defaultdict(int)
        # Base case
        for i in range(len(e_sent)):
            V[i,0] = pi[e_sent[i]][f_sent[0]]

        # Recursive case
        for j in range(1,len(f_sent)):
            for i in range(1,len(e_sent)):
                maximum = 0

                # Loop over previous possible alignments_pred
                for ii in range(len(e_sent)):
                    if i-ii != 0 and len(e_sent)-1 > 1:
                        p = V[ii,j-1] * q[len(e_sent)-1][i-ii] * pi[e_sent[i]][f_sent[j]]

                    # Update p if highest value
                    if p > maximum:
                        maximum = p
                        step = ((i,j),(ii,j-1))
                        trace[step[0]] = step[1]
                        V[i,j] = maximum

        # highest probability last column
        i = np.argmax(V[:,-1])
        # Last French word alignment
        alignments.append((str(sen_count).zfill(4), i, len(f_sent)-1))
        # Backtrack from last French word
        for j in reversed(range(1,len(f_sent))):
            prev_pos = trace[(i,j)]
            alignments.append((str(sen_count).zfill(4), prev_pos[0], prev_pos[1]))

        # print(alignments)

    return alignments

def output_naacl(alignments, fn):
    with open(fn, 'w') as fp:
        fp.write('\n'.join('%s %s %s S' % x for x in alignments))
