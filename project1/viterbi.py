from load_corpus import load_train
from lexicon import init_lexicon

#@TODO Add Sure / Probable (S / P) label after alignment. How?

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

def output_naacl(alignments, fn):
    with open(fn, 'w') as fp:
        fp.write('\n'.join('%s %s %s S' % x for x in alignments))
