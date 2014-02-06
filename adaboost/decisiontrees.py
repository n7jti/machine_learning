#!/usr/bin/python

import numpy as np
import argparse
import math

rawData = [[24, 40000, True],
        [53, 52000, False],
        [23, 25000, False],
        [25, 77000, True],
        [32, 48000, True],
        [52, 110000,True],
        [22, 38000, True],
        [43, 44000, False],
        [52, 27000, False],
        [48, 65000, True]
       ]; 

def PDegree(bool, data):
  count = 0;
  for i in range(len(data)):
    if data[i][2] == bool:
      count+=1
  return float(count) / len(data);


def etaX (data):
  sum = 0
  sum += PDegree(True, data) * math.log(PDegree(True, data),2)
  sum += PDegree(False, data) * math.log(PDegree(False, data),2)
  return -sum


def split (y, idx, data):
  left = []
  right = []
  for i in range(len(data)) in :
    if data[i][idx] <= y:
      left.append(data[i])
    else:
      right.append(data[i])

  return left, right

def pxgylte (bool, val, idx, data):
  countCorrect = 0
  total = 0

  for i in range(len(data)):
    if data[i][idx] <= val:
      total += 1
      if data[i][2] == bool:
        countCorrect += 1

  return float(countCorrect) / total;

def main ():
  print "len", len(rawData)
  print "PDegree(True)", PDegree(True, rawData)
  print "H(X)", etaX(rawData);

  print "P(True | y <= 30 )", pxgylte(True, 30, 0, rawData)

if __name__ == "__main__":
    main()
