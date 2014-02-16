#!/usr/bin/python

import argparse
import numpy as np
from sklearn import cross_validation
from sklearn import svm

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

  # X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y[:,1], test_size=0.4, random_state=0)

  
  clf = svm.SVC(kernel='linear', C=1)
  scores = cross_validation.cross_val_score(clf, x, y[:,0], cv=5)

  print clf
  print "coef", clf.coef_
  print "intercept", clf.intercept_
  print scores

  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == "__main__":
    main()
