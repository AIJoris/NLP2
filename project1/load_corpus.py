## Loads training corpus
def load_train(fpath):
    with open(fpath+'/training/hansards.36.2.e','r') as f:
        english = [x.split() for x in f.read().splitlines()]

    with open(fpath+'/training/hansards.36.2.f','r') as f:
        french = [x.split() for x in f.read().splitlines()]


    [x.insert(0,"-NULL-") for x in french]

    return [english,french]
