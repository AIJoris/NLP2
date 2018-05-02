import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")

def make_plots(perplex_fn, aer_fn):


    perplexity = pickle.load(open( perplex_fn, "rb" ))
    aer = [float(x.split()[-1].decode("utf-8")) for x in pickle.load(open( aer_fn, "rb" ))]


    plt.plot(np.arange(1,len(aer)+1), aer)
    plt.xlabel("Number of iterations")
    plt.ylabel("AER")
    plt.show()

    plt.plot(np.arange(1,len(perplexity)+1), perplexity)
    plt.xlabel("Number of iterations")
    plt.ylabel("Perplexity")
    plt.show()
