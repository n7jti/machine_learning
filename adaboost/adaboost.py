#!/usr/bin/python
from scipy import *
import scipy.sparse as sp
import dstump as ds
import pylab as pl
import numpy as np
import adaboost_test as test
import operator

def adaboost_train (x, y, T):
  cf = x.shape[1]
  n = y.shape[0]
  weights = ones(n)/n
  pplus = sum(weights * (y > 0))

  H = []
  A = []
  I = []
  TE = []

  for t in range(T):

    # Let's train on all the features and find the one that works the best
    decisionVariables = []
    score = []
    we = []
    for idx in range(cf):
      f = x[:,idx]

      # train the stump
      (dv, err) = ds.stump_fit(f, y, weights, pplus)
      decisionVariables.append(dv)
      we.append(err)

      # score the classifiers on all features for this round
      score.append(abs(.5-we[idx]))
      # print "err", err, "score",score[idx]  

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

    idx = min(enumerate(HE), key=operator.itemgetter(1))[0]

    return HE, idx

def main ():

  #
  #  Train
  #

  gen = test.four_clusters
  (x, y) = gen(500)

  n = y.shape[0]
  T = 30
  A, H, I , TE = adaboost_train (x, y, T)

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

  #
  # Validate
  #

  (xv, yv) = gen(500)
  HE, t = adaboost_find_t(A,H,I,xv, yv)

  print "t", t

  pl.clf()
  pl.plot(HE,'o')
  pl.show()

  #
  # Test
  #

  (xt, yt) = gen(500)
  y_hat = adaboost_predict(A,H,I,xt,t)
  err = (y_hat * yt < 0).sum() / float(yt.shape[0])
  print "err", err

if __name__ == "__main__":
    main()

