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
    for col in range(cols):
      c = 2 * (X[:,col] * (t-(w0 + np.dot(X,w) - (w[col] * X[:,col]) ))).sum()
      w[col] = soft(a[col],c,l)
    w0 = ((t - np.dot(X,w)).sum())/rows
    delta = linalg.norm(old-w, np.inf);
    if (delta <= d):
      break;
  return (w,w0)

def lambda_max (X, w, w0, y):
    return 2 * linalg.norm(np.dot(X.T,(y-(np.dot(X,w) + w0))),np.inf)

def countNonZeros (w):
  cnt = w.shape[0]
  total = 0;

  for i in range(cnt):
    if w[i] != 0:
      total += 1
  return total

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


  #
  # Star
  #

  # Load a text file of integers:
  sLbl = np.loadtxt("data/star_labels.txt", dtype=np.int)

  # Load a text file of strings:
  sFeat = open("data/star_features.txt").read().splitlines()

  # Load a matrix market matrix, convert it to csc format:
  sData = io.mmread("data/star_data.mtx").tocsc()


  return uvLbl, uvFeat, uvData, sLbl, sFeat, sData

def main ():
  uvLbl, uvFeat, uvData, sLbl, sFeat, sData = loadData()




if __name__ == "__main__":
    main()
