#!/usr/bin/python

import argparse
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def loadData (subdir):
  # Load a csv of floats:
  data = np.genfromtxt(subdir + '/data.csv', delimiter=",", skip_header=0)
  labels = np.genfromtxt(subdir + '/labels.csv', delimiter=",", skip_header=0)
  
  return data, labels 

def main ():
  parser = argparse.ArgumentParser(description='Perform cross validation on the dataset')
  parser.add_argument('-source', help='subdirectory for the source data')
  args = parser.parse_args()

  # Load the dataset
  x, y = loadData( args.source )

  
  # Split the dataset in two equal parts
  X_train, X_test, y_train, y_test = train_test_split(x, y[:,1], test_size=0.5, random_state=0)

  param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   ]

  scores = ['precision', 'recall']

  for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print

    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring=score)
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
