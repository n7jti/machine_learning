#!/usr/bin/python

import numpy as np
import argparse


#
# Kernalized Perceptron Algorithym
#

def sign (x):
  if (x >= 0):
    return 1
  return -1

def ptronStep (t, M, x, y, k):
  sum = 0
  for j in M:
    sum += y[j] * k(x[j],x[t])
  y_hat = sign(sum)
  return y_hat

def ptron (x, y, k):
  M = []
  L = []

  # start the weights at zero
  w = np.zeros(x.shape[1])

  # Do this for all rows in our data
  for i in range(x.shape[0]):

    # get the prediction
    y_hat = ptronStep(i, M, x, y, k)

    #if we made a mistake, note the index and update the weights
    if y_hat != y[i]:
      M.append(i)
      w = w + y[i] * x[i]
    
    #print out our loss function every 100 steps
    if (i + 1) % 100 == 0:
      avg = len(M)/float(i+ 1)
      print i+1, ",", avg
      L.append([i+1,avg])

  return w,L

#
# Kernels
#

def k1p (u, v):
  return u.dot(v) + 1


#
# Main
#

def loadData ():
  # Load a csv of floats:
  testData = np.genfromtxt("test.csv", delimiter=",", skip_header=1)
  y = testData[:,0]
  x = testData[:,1:]
  return x,y
 
def main ():
  parser = argparse.ArgumentParser(description='Kernalized Perceptron!')
  parser.add_argument('-o', help='output file to write Loss Function', default='out.csv')
  args = parser.parse_args()

  x,y = loadData()
  w, L = ptron(x,y,k1p)

  np.savetxt(args.o, L, delimiter=",")


if __name__ == "__main__":
    main()
