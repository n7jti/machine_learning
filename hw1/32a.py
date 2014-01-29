#!/usr/bin/python

import numpy as np
from scipy import linalg
import math

# Soft thresholding function
def soft(a,c,l):
  if c < -l:
    return (c + l)/a
  elif c > l:
    return (c-l)/a
  else:
    return 0.0

def gradientDescent(X, w, w0, t, l, d):
  # calculate a's once
  rows = X.shape[0]
  cols = X.shape[1]
  a = 2 * ((X * X).sum(axis=0));

  while True:
    old = np.array(w)

    for col in range(cols):
      c = 2 * (X[:,col] * (t-(w0 + np.dot(X,w) - (w[col] * X[:,col]) ))).sum(axis=0)
      #print c
      w[col] = soft(a[col],c,l)
    
    # compute the updated w0
    w0 = ((t - np.dot(X,w)).sum())/rows

    delta = linalg.norm(old-w, np.inf);

    if (delta <= d):
      break;
  return (w,w0)

def checkValue (X, w, w0, t, l ):
  print "l    ", l
  print "w0   ", w0
  print "w    ", w
  print "check", np.dot(2*X.transpose(), np.dot(X,w) + w0 * np.ones(X.shape[0]) - t)


def syntheticData (N, d, k, sigma):

  # get our random matrix
  X = sigma * np.random.randn(N, d)

  # get our w0
  w0 = 0;

  # Create our weight vector with the firt k element having a weight of either +10 or -10
  w=np.zeros(d)
  sign = 1;
  for i in range(k):
    w[i] = sign * 10
    sign *= -1

  # generate a vector consisting of random noise!
  epsilon = sigma * np.random.randn(N)

  # generate our synthetic "truth" vector
  y = np.dot(X,w) + w0 + epsilon

  return X,w,w0,y

def initial_w(X, l, y):
  return np.dot(np.dot(linalg.inv(np.dot(X.T,X) + l * np.identity(X.shape[1])),X.T),y)

#def lambda_max (X, y):
#  return 2 * linalg.norm(np.dot(X.T,(y-np.average(y))),np.inf)


def lambda_max (X, w, w0, y):
    return 2 * linalg.norm(np.dot(X.T,(y-(np.dot(X,w) + w0))),np.inf)

def precision (w, k):
  cnt = w.shape[0]
  correct = 0;
  total = 0;

  for i in range(cnt):
    if w[i] != 0:
      total += 1
      if i < k:
        correct += 1

  return correct/float(total)

def recall (w,k):
  correct = 0
  total = 0

  for i in range(k):
    if w[i] != 0:
      total += 1

  return total/float(k)

def countNonZeros (w):
  cnt = w.shape[0]
  total = 0;

  for i in range(cnt):
    if w[i] != 0:
      total += 1
  return total

def main ():
  rows = 50
  cols = 75
  nonzeros = 5
  sigma = 1


  X,wt,w0t,y  = syntheticData(rows, cols, nonzeros, sigma)

  w = np.zeros(75);
  w0 = np.average(y);
  l=lambda_max(X,w,w0,y)

  print "lambda,", "precision,","recall"
  while l > 1:
    w,w0 = gradientDescent(X, w, w0, y, l, .0001)
    cnt = countNonZeros(w)
    if cnt > 0:
      print l,",", precision( w, nonzeros), ",", recall(w, nonzeros)
    l *= .8
  

if __name__ == "__main__":
    main()
