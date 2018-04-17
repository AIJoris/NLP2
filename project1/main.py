from load_corpus import load_train

## Load parallel corpus
[english,french] = load_train('data')

## Create lexicon (double dict)
# lexicon = create_lexicon(english,french)
# lexicon[word_e][word_f] = prob

## Estimate translation sentence length
