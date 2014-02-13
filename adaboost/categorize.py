#!/usr/bin/python2
from scipy import *
import scipy.sparse as sp
import scipy.linalg as la
#See http://scikit-learn.org/stable/modules/feature_extraction.html
import sklearn.feature_extraction as fe
import tok
import dstump as ds
import pylab as pl
import numpy as np
import operator
from datetime import datetime
import os

def adaboost_predict (A, H, I, x, t):
  n=x.shape[0]
  out = np.zeros(n)
  for i in range(t):
    out +=  A[i] * ds.stump_predict(x[:,I[i]], H[i])

  return np.sign(out);

def getNewsgroups (dirname = "20_newsgroups/"):
  files = []
  for fname in os.listdir(dirname):
    if os.path.isdir(dirname + fname):
      files.append(fname)

  return files


def load ():

  S = np.loadtxt("matrix1.out")

  A,H,I = np.vsplit(S,3)

  A = A.flatten()
  H = H.flatten()
  I = I.flatten()

  return A, H, I

def main ():
  #Read text, try removing comments, headers ... See tok.py for implementation.
  corpus = tok.fill_corpus(["alt.atheism", "comp.windows.x"])
  #corpus = tok.fill_corpus(["alt.atheism", "soc.religion.christian"])


  #Create training data
  ctr = reduce(list.__add__, map(lambda x: x[:600], corpus))
  ytr = zeros(len(ctr)); ytr[:600] = -1; ytr[600:] = 1  

  #Train a bag-of-words feature extractor.
  #You're free to play with the parameters of fe.text.TfidfVectorizer, but your answers
  #*should be* answered for the parameters given here. You can find out more about these
  #on the scikits-learn documentation site.
  tfidf = fe.text.TfidfVectorizer(min_df=5, ngram_range=(1, 4), use_idf=True, encoding="ascii")

  #Train the tokenizer.
  ftr = tfidf.fit_transform(ctr)
  ftr = ftr.tocsc()

  #This maps features back to their text.
  feature_names = tfidf.get_feature_names()

  #tokenizer is trained, then 
  A, H , I = load()

  newsgroups = getNewsgroups()
  newsgroups = sort(newsgroups)

  for group in newsgroups:
    bag = tok.fill_corpus([group])
    bag = bag[0]
    f = tfidf.transform(bag).tocsc()
    y_hat = adaboost_predict(A,H,I,f,len(A))

    left = (y_hat > 0).sum() / float(len(y_hat))
    if (left > .5):
      print group, "comp.windows.x("+str(left)+")"
    else:
      print group, "alt.atheism("+str(1-left)+")"

if __name__ == "__main__":
  main()
