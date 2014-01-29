#!/usr/bin/python

import numpy as np
from scipy import linalg
import math
import argparse
import time
import sys
import scipy.io as io
import scipy.sparse as sparse

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
  c= 2 * ( np.dot(X.T,(t - (np.dot(X,w) + w0))) ) + w * a

  while True:
    old = np.array(w)

    for col in range(cols):
      #co = 2 * (X[:,col] * (t-(w0 + np.dot(X,w) - (w[col] * X[:,col]) ))).sum()
      #print "co - c[cols]", co - c[col]

      wold = np.array(w)
      w[col] = soft(a[col],c[col],l)
      dw = w - wold

      #c = c - 2 * np.dot(X.T, np.dot(X,dw)      ) + dw * a
      c = c - 2 * np.dot(X.T, X[:,col] * dw[col]) + dw * a

    w0old = np.array(w0)
    w0 = ((t - np.dot(X,w)).sum())/rows
    dw0 = (w0 - w0old) * np.ones(X.shape[0])

    c = c - 2 * np.dot(X.T,dw0)

    delta = linalg.norm(old - w, np.inf)
    if ( delta <= d ):
      break;

  return (w,w0)

def lambdaMax (X, w, w0, y):
    return 2 * linalg.norm(np.dot(X.T,(y-(np.dot(X,w) + w0))),np.inf)

def countNonZeros (w):
  cnt = w.shape[0]
  total = 0;

  for i in range(cnt):
    if w[i] != 0:
      total += 1
  return total

def RMS (X, w, w0, y):
  err = y - (w0 + np.dot(X,w))
  return np.sqrt(np.square(err).sum() / float(err.shape[0]))

def ssd (u,v):
  return np.square(u - v).sum()

def loadData ():

  #
  #  upvote
  #

  # Load a text file of integers:
  uvLbl = np.loadtxt("data/upvote_labels.txt", dtype=np.int)

  # Load a text file of strings:
  uvFeat = open("data/upvote_features.txt").read().splitlines()

  # Load a csv of floats:
  uvData = np.genfromtxt("data/upvote_data.csv", delimiter=",")


  return uvLbl, uvFeat, uvData

def main ():
  print "Loading Data"
  start = time.clock()
  uvLbl, uvFeat, uvData = loadData()
  stop = time.clock()

  uvLblTrain = uvLbl[0:4000]
  uvLblValidate = uvLbl[4000:5000]
  uvLblTest = uvLbl[5000:]

  uvDataTrain = uvData[0:4000,:]
  uvDataValidate = uvData[4000:5000,:]
  uvDataTest = uvData[5000:,:]

  print "Loaded", stop - start


  w0 = np.average(uvLblTrain)
  w = np.zeros(uvDataTrain.shape[1])
  #l=lambdaMax(uvDataTrain, w, w0, uvLblTrain);
  l = 5.13431277185 

  print "time, lambda, RMS Train, RMS Validate, RMS Test, nonzeros"

  start = time.clock()
  w,w0 = gradientDescent(uvDataTrain, w, w0, uvLblTrain, l, .001)
  stop = time.clock()
  cnt = countNonZeros(w)
  sys.stdout.flush()
  print stop - start, "," ,l ,"," , RMS(uvDataTrain,w,w0,uvLblTrain), "," , RMS(uvDataValidate,w,w0,uvLblValidate), ",", RMS(uvDataTest,w,w0,uvLblTest), ",", countNonZeros(w)
  sys.stdout.flush()

  print "w"
  print w

if __name__ == "__main__":
    main()
