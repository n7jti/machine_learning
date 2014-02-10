#!/usr/bin/python

import numpy as np
from scipy import linalg
import argparse
import math
import operator

rawData = [[24, 40000, 1],
        [53, 52000, -1],
        [23, 25000, -1],
        [25, 77000, 1],
        [32, 48000, 1],
        [52, 110000,1],
        [22, 38000, 1],
        [43, 44000, -1],
        [52, 27000, -1],
        [48, 65000, 1]
       ] 

testData= [
            [ 1, 1, 1],
            [ 1,-1, 1],
            [ 1, 1, 1],
            [ 1,-1, 1],
            [-1, 1, 1],
            [-1,-1,-1],
          ]

def Px(x, data):
  count = 0;
  for i in range(len(data)):
    if data[i][2] == x:
      count+=1
  return float(count) / len(data);


def etaX (data):
  valList = [1, -1]
  sum = 0
  for i in range(len(valList)):
    px = Px(valList[i], data)
    if px != 0:
      sum += px * math.log(px,2)
  return -sum

def etaSplit (y, idx, data):
  left, right = split(y, idx, data)

  if len(left) == 0 or len(right)==0:
    return etaX(data)

  pLeft = float(len(left))/len(data)
  pRight = float(len(right))/len(data)

  pLeftTrue = Px(1, left)
  pLeftFalse = Px(-1,left)

  pRightTrue = Px(1, right)
  pRightFalse = Px(-1, right)

  sum = 0
  if pLeftTrue != 0:
    sum += pLeftTrue * math.log(pLeftTrue,2)
  if pLeftFalse != 0:
    sum += pLeftFalse * math.log(pLeftFalse,2)
  outsum = pLeft * sum

  sum = 0
  if pRightTrue != 0:
    sum += pRightTrue * math.log(pRightTrue,2)
  if pRightFalse != 0:
    sum += pRightFalse * math.log(pRightFalse,2)
  outsum += pRight * sum

  return - outsum


def etaSplitl (w, data):
  left, right = splitl(w, data)

  if len(left) == 0 or len(right)==0:
    return etaX(data)

  pLeft = float(len(left))/len(data)
  pRight = float(len(right))/len(data)

  pLeftTrue = Px(1, left)
  pLeftFalse = Px(-1,left)

  pRightTrue = Px(1, right)
  pRightFalse = Px(-1, right)

  sum = 0
  if pLeftTrue != 0:
    sum += pLeftTrue * math.log(pLeftTrue,2)
  if pLeftFalse != 0:
    sum += pLeftFalse * math.log(pLeftFalse,2)
  outsum = pLeft * sum

  sum = 0
  if pRightTrue != 0:
    sum += pRightTrue * math.log(pRightTrue,2)
  if pRightFalse != 0:
    sum += pRightFalse * math.log(pRightFalse,2)
  outsum += pRight * sum

  return - outsum

def split (y, idx, data):
  left = []
  right = []
  for i in range(len(data)):
    if data[i][idx] <= y:
      left.append(data[i])
    else:
      right.append(data[i])

  return left, right

def splitl (w, data):
  left = []
  right = []

  x = np.array(data)[:,0:2]

  for i in range(len(data)):
    if np.sign(w.dot(x[i,:])-1) > 0:
      left.append(data[i])
    else:
      right.append(data[i])

  return left, right


def pxgylte (x, val, idx, data):
  countCorrect = 0
  total = 0

  for i in range(len(data)):
    if data[i][idx] <= val:
      total += 1
      if data[i][2] == x:
        countCorrect += 1

  return float(countCorrect) / total;

def pxgygt (x, val, idx, data):
  countCorrect = 0
  total = 0

  for i in range(len(data)):
    if data[i][idx] > val:
      total += 1
      if data[i][2] == x:
        countCorrect += 1

  return float(countCorrect) / total;

 
def ig (y, idx, data):
  return etaX(data)-etaSplit(y,idx,data)

def igl (w, data):
  return etaX(data)-etaSplitl(w,data)

def learn(data):
  # I'm going to use linnear regression as a classifier
  x = np.array(data)[:,0:2]
  y = np.array(data)[:,2]
  print "x", x
  print "y", y
  w = linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
  w = w / .1
  print w
  for line in range(x.shape[0]):
    print np.sign(w.dot(x[line,:])-1)
  return w


def main ():
  w=learn(rawData)
  print "etaX(data)", etaX(rawData)
  print "etaSplitl(w,data)", etaSplitl(w,rawData)
  print "ig(w,data)", igl(w, rawData)


  



if __name__ == "__main__":
    main()

