from load_corpus import load_train
from lexicon import init_t_e_f

## Load parallel corpus
[english,french] = load_train('data')



## Create uniform t(e|f)
p_t_e = init_t_e_f(english, french)

## Estimate translation sentence length
