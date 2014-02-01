#!/usr/bin/python

import numpy as np
import argparse
import math

#
# SVN
#

def sign (x):
  if (x >= 0):
    return 1
  return -1

def svn (x, y, eta, C):
  M = []
  L = []

  # start the weights at zero
  w = np.zeros(x.shape[1])
  w0 = 0

  # Do this for all rows in our data
  for i in range(x.shape[0]):
    t=i+1
    etat = eta / math.sqrt(t)

    #if we made a mistake, note the index and update the weights

    if y[i] * ( np.dot(w,x[i,:]) + w0) <= 1:
      # Mistake Case
      M.append(i)
      w = w - etat * ( w - C * y[i] * x[i,:])
      w0 = w0 + etat * (C * y[i])
    else:
      w = w - etat * ( w )
      
    #print out our loss function every 100 steps
    if (t) % 100 == 0:
      avg = len(M)/float(t)
      print t, ",", avg
      L.append([t,avg])

  return w, w0, L

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
  eta = 1.0E-5
  C = 1
  w, w0, L = svn(x,y,eta, C)

  np.savetxt(args.o, L, delimiter=",")


if __name__ == "__main__":
    main()
