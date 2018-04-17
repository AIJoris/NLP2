## Loads training corpus
def load_train(fpath):
    with open(fpath+'/training/hansards.36.2.e','r') as f:
        english = f.read().splitlines()

    with open(fpath+'/training/hansards.36.2.f','r') as f:
        french = f.read().splitlines()
    return [english,french]
