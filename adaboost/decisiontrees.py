#!/usr/bin/python

import numpy as np
import argparse
import math
import operator

rawData = np.array([[24, 40000, 1],
        [53, 52000, -1],
        [23, 25000, -1],
        [25, 77000, 1],
        [32, 48000, 1],
        [52, 110000,1],
        [22, 38000, 1],
        [43, 44000, -1],
        [52, 27000, -1],
        [48, 65000, 1]
       ])

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


def split (y, idx, data):
  left = []
  right = []
  for i in range(len(data)):
    if data[i][idx] <= y:
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

def stuff ():
  print "First Split, Slalary <= 27000 :", ig(27000,1,rawData)
  l,r = split(27000, 1, rawData)

  print "l is done, all FALSE!"
  print "l", l

  print "Second split on r, age<=32", ig(32,0,r)
  rl, rr = split(32,0,r)

  print "rl is done, all TRUE"
  print "rl", rl

  print "lots of choices for next split.  Choosing to split on salary <=44000", ig(44000,1,rr)
  rrl, rrr = split(44000,1,rr);

  print "rrl is done, all FALSE"
  print "rrl", rrl
  
  print "lots of choices again for next split. Choosing to split on Age <= 52", ig(52, 0, rrr)
  rrrl, rrrr = split(52, 0, rrr);
  print "rrrl", rrrl
  print "rrrr", rrrr

def main ():
  stuff()
  rawData.view('i8,i8,i8').sort(order=['f0','f1'],axis=0)
  #print "rawData", rawData

  for idx in range(rawData.shape[0]) :
    val = rawData[idx,0]
    print "age", val, "ig", ig(val,0,rawData)

  rawData.view('i8,i8,i8').sort(order=['f1','f0'],axis=0)
  for idx in range(rawData.shape[0]) :
    val = rawData[idx,1]
    print "salary", val, "ig", ig(val,1,rawData)

  
  

if __name__ == "__main__":
    main()
