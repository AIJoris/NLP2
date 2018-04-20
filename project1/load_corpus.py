from collections import Counter

## Loads training corpus
def load_train(fpath, type):
    if type == 'train':
        fn_e = '/training/hansards.36.2.e'
        fn_f = '/training/hansards.36.2.f'
    elif type == 'test':
        fn_e = '/testing/test/test.e'
        fn_f = '/testing/test/test.f'
    elif type == 'val':
        fn_e = '/validation/dev.e'
        fn_f = '/validation/dev.f'

    with open(fpath+fn_e,'r') as f:
        english = [x.split() for x in f.read().splitlines()]

    with open(fpath+fn_f,'r') as f:
        french = [x.split() for x in f.read().splitlines()]

    [x.insert(0,"-NULL-") for x in english]

    return [english,french]

def count_words(sentences):
    words = []
    for s in sentences:
        for w in s:
            words.append(w)

    return Counter(words)

def replace_singletons(sentences, count):
    updated_sentences = []
    for s in sentences:
        sentence = []
        for w in s:
            if count[w] > 1: sentence.append(w)
            else: sentence.append('-LOW-')
        updated_sentences.append(sentence)
    return updated_sentences
