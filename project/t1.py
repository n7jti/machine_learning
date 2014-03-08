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
  parser.add_argument('-source', help='subdirectory for the source data', default='train2')
  parser.add_argument('-prefix', help='prefix for file names', default='five')
  parser.add_argument('-index' , help='index of the day to train for', default='1')
  args = parser.parse_args()

  x, y = loadData( args.source, args.prefix )

  
  # Split the dataset in two equal parts
  X_train, X_test, y_train, y_test = train_test_split(x, y[:,int(args.index)], test_size=0.5, random_state=0)


  # Set the parameters by cross-validation
  #for i in range(21): C.append(10.0**(i-5))
  #for i in range(17): gamma.append(10**(i-14))

  #C=[0.001,0.01,0,1,1,10,100,1000,10000,100000]
  #gamma=[1.0e-7,1.0e-6,1.0e-5,1.0e-4,0.001,0.01,0.1,1,10,100]

  C=[100000,1000000,10000000]
  gamma=[1.0e-6,1e-7,1.0e-8]

  tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': C}]

  print("# Tuning hyper-parameters for accuracy")
  clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
  clf.fit(X_train, y_train)

  print "Best parameters set found on development set:"
  print
  print clf.best_estimator_
  print
  print "Grid scores on development set:"
  print
  for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
  print

  print "Detailed classification report:"
  print
  print "The model is trained on the full development set."
  print "The scores are computed on the full evaluation set."
  print

  y_true, y_pred = y_test, clf.predict(X_test)
  print classification_report(y_true, y_pred)
  print

if __name__ == "__main__":
    main()
