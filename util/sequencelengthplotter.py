'''
This python program reads in all the source text data, determining the length of each
file and plots them in a histogram.

This is used to help determine the bucket sizes to be used in the main program.
'''

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os

dirs = ["data/aclImdb/test/pos", "data/aclImdb/test/neg", "data/aclImdb/train/pos", "data/aclImdb/train/neg"]

def main():
    lengths = []
    count = 0
    for d in dirs:
        print "Grabbing sequence lengths from: {0}".format(d)
        for f in os.listdir(d):
            count += 1
            if count % 100 == 0:
                print "Determining length of: {0}".format(f)
            with open(os.path.join(d, f), 'r') as review:
                tokens = tokenize(review.read().lower())
                numTokens = len(tokens)
                if numTokens not in lengths:
                    lengths.append(numTokens)
                else:
                    lengths.append(numTokens)

    #mu = np.mean(lengths)
    #sigma = np.std(lengths)
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    num_bins = 50
    n, bins, patches = plt.hist(x,  num_bins, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title("Frequency of Sequence Lengths")
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
    plt.xlim(0,1500)
    plt.show()

'''
This function tokenizes sentences
'''
def tokenize(text):
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)


if __name__=="__main__":
    main()
