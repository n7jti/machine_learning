#!/usr/bin/python

import numpy as np
from scipy import linalg

# Soft thresholding function
def soft(a,c,l):
  if c < -l:
    return (c + l)/a
  elif c > l:
    return (c-l)/a
  else:
    return 0.0

# this calculates the innermost term of the c calculation
def innerTermC(X, w, curRow, curCol):
  term = 0
  for col in range(X.shape[1]):
    if col != curCol:
      term += w[col] * X[curRow, col]
  return term

# this calculate the innermot term of the W0 calcuation
def innerTermW0(X, w, curRow):
  term = 0
  for col in range(X.shape[1]):
    term += w[col] * X[curRow, col]
  return term

def gradientDescent(X, w, w0, t, l, d):
  while True:
    old = np.array(w)
   
    # compute the updated w
    for col in range(X.shape[1]):
      a=0
      c=0
      for row in range(X.shape[0]):
        a += X[row,col] * X[row,col]
        c += X[row,col] * (t[row] - (w0 + innerTermC(X, w, row, col)))

      a *= 2
      c *= 2

      w[col] = soft(a,c,l)

      #print "c:",c
    
    # compute the updated w0
    w0 = 0
    #print "X.shape",X.shape
    for row in range(X.shape[0]):
      w0 += t[row] - innerTermW0(X,w,row)
    w0 = w0 / X.shape[1]

    delta = linalg.norm(old-w,np.inf);
    #print "delta", delta

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


def main ():
  
  X,wt,w0t,y  = syntheticData(50, 75, 5, 1)
  l=200
  w = np.zeros(75);
  w0 = np.average(y);
  w,w0 = gradientDescent(X, w, w0, y, l, .001)

  checkValue(X, w, w0, y, l)

  #print "w", w
  #print "w0", w0


if __name__ == "__main__":
    main()
