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

  while True:
    old = np.array(w)
    mydot = np.dot(X,w)
    for col in range(cols):
      mydot -= (w[col] * X[:,col])
      c = 2 * (X[:,col] * (t-(w0 + mydot) )).sum()
      w[col] = soft(a[col],c,l)
      mydot += (w[col] * X[:,col])

    w0 = ((t - np.dot(X,w)).sum())/rows
    delta = linalg.norm(old-w, np.inf);
    if (delta <= d):
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
  return np.sqrt(np.square(err) / float(err.shape[0]))
  return np.norm(y - (w0 + np.dot(X,w))) / float(y.shape[0])

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
  uvLbl, uvFeat, uvData = loadData()

  uvLblTrain = uvLbl[0:4000]
  uvLblValidate = uvLbl[4000:5000]

  uvDataTrain = uvData[0:4000,:]
  uvDataValidate = uvData[4000:5000,:]

  print "Loaded"


  w0 = np.average(uvLblTrain)
  w = np.zeros(uvDataTrain.shape[1])
  l=lambdaMax(uvDataTrain, w, w0, uvLblTrain);

  print "lambda, RMS Train, RMS Validate"
  while l > 1:

    #Learn some parmeters
    w,w0 = gradientDescent(uvDataTrain, w, w0, uvLblTrain, l, .001)
    cnt = countNonZeros(w)
    if cnt > 0:
      #Now, look at the RSS on the validation data

      print l,",", RMS(uvDataTrain,w,w0,uvLblTrain), "," , RMS(uvDataValidate,w,w0,uvLblValidate)
      sys.stdout.flush()
    l *= .8





if __name__ == "__main__":
    main()
