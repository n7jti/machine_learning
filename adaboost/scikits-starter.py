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


def adaboost_train (x, y, T):
  cf = x.shape[1]
  n = y.shape[0]
  weights = ones(n)/n

  H = []
  A = []
  I = []
  TE = []

  for t in range(T):
    pplus = sum(weights * (y > 0))

    # Let's train on all the features and find the one that works the best
    decisionVariables = []
    score = []
    we = []
    for idx in range(cf):
      f = x[:,idx]

      # train the stump
      (dv, err) = ds.stump_fit(f, y, weights, pplus)
      we.append( err )
      decisionVariables.append(dv)

      # score the classifiers on all features for this round
      score.append(abs(.5-err))
     
    print "Round: ", t, str(datetime.now())
    # choose the one feature we'll use for this round's classifier
    I.append(np.argmax(score))
    H.append(decisionVariables[I[t]])
    eps = we[I[t]]
    
    # calculate our alpha
    A.append(.5 * math.log((1-eps)/eps))

    # update the weights
    numerators = weights * np.exp( -A[t] *  y * ds.stump_predict(x[:,I[t]], H[t]) )
    Z = numerators.sum()
    weights = numerators / Z

    # Calculate the overall training errors
    y_hat = adaboost_predict(A,H,I,x, len(A))
    TE.append((y_hat * y < 0).sum() / float(n))

  return A, H, I, TE


def adaboost_predict (A, H, I, x, t):
  n=x.shape[0]
  out = np.zeros(n)
  for i in range(t):
    out +=  A[i] * ds.stump_predict(x[:,I[i]], H[i])

  return np.sign(out);


def adaboost_find_t (A, H, I, x, y):
    n=x.shape[0]
    out = np.zeros(n)
    t=len(A)
    HE = []
    for i in range(t):
      out +=  A[i] * ds.stump_predict(x[:,I[i]], H[i])
      HE.append( (np.sign(out) * y < 0).sum() / float(n) )

    idx = np.argmin(HE)

    return HE, idx + 1

def main ():
  #Read text, try removing comments, headers ... See tok.py for implementation.
  #corpus = tok.fill_corpus(["alt.atheism", "comp.windows.x"])
  corpus = tok.fill_corpus(["alt.atheism", "soc.religion.christian"])

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

  m = 30
  #This shouldn't take more than 20m.
  A, H, I , TE = adaboost_train(ftr, ytr, m)

  for i in range(m):
    print "T", i, "index:", I[i], "feature name:", feature_names[I[i]]

  # Plot
  pl.subplot(2,1,1)
  pl.xlabel('steps of adaboost')
  pl.ylabel('magnitude of alpha')
  pl.plot(np.abs(A),'o')
  pl.subplot(2,1,2)
  #pl.axis([0,50,0,.5])
  pl.xlabel('steps of adaboost')
  pl.ylabel('training error')
  pl.plot(TE,'o')
  pl.show()


  #Create validation data
  cva = reduce(list.__add__, map(lambda x: x[600:800], corpus))
  yva = zeros(len(cva)); yva[:200] = -1; yva[200:] = 1

  #tfidf tokenizer is not trained here.
  fva = tfidf.transform(cva).tocsc()

  #<Validation code goes here>
  HE, t = adaboost_find_t(A,H,I,fva, yva)

  print "t", t
  A = A[:t]
  H = H[:t]
  I = I[:t]

  S = np.vstack((A,H,I))
  np.savetxt("matrix2.out", S);

  pl.clf()
  pl.plot(HE,'o')
  pl.show()

  #Create test data
  #Some lists have less than a thousand mails. You may have to change this.
  cte = reduce(list.__add__, map(lambda x: x[800:], corpus))
  yte = zeros(len(cte)); yte[:200] = -1; yte[200:] = 1

  fte = tfidf.transform(cte).tocsc()

  #<Testing code goes here>
  y_hat = adaboost_predict(A,H,I,fte,t)
  err = (y_hat * yte < 0).sum() / float(yte.shape[0])
  print "err", err


if __name__ == "__main__":
    main()
