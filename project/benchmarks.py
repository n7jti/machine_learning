#!/usr/bin/python

import argparse
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def loadData (subdir, prefix):
  # Load a csv of floats:
  data = np.genfromtxt(subdir + '/' + prefix + 'data.csv', delimiter=",", skip_header=0)
  labels = np.genfromtxt(subdir + '/' + prefix + 'labels.csv', delimiter=",", skip_header=0)
  
  return data, labels 

def main ():
  parser = argparse.ArgumentParser(description='Perform cross validation on the dataset')
  parser.add_argument('-source', help='subdirectory for the source data')
  parser.add_argument('-prefix', help='prefix for file names')
  args = parser.parse_args()

  x, y = loadData( args.source, args.prefix )

  y_true = y[:,0]
  y_rain = np.ones(y_true.shape[0])
  y_dry = y_rain * -1
  print 'Benchmarks:' 
  print
  print 'wet'
  print classification_report(y_true, y_rain)
  print
  print 'dry'
  print classification_report(y_true, y_dry)
  print

  for i in range(5):
      y_true = y[:,i]
      y_yesterday = x[:,9]
      print 'Benchmarks:',i 
      print 'yesterday'
      print classification_report(y_true, y_yesterday)
      print
  

if __name__ == "__main__":
    main()
