#!/usr/bin/python
from scipy import *
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

def load ():
      # Load a csv of floats:
      train = np.genfromtxt("data/train.csv", delimiter=",", skip_header=1)
      y_train = train[:,0]
      x_train = train[:,1:]

      test = np.genfromtxt("data/test.csv", delimiter=",", skip_header=1)
      x_test = test
      
      return y_train,x_train, x_test
def main ():
    y, x, x_test = load();

    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_classes = 10

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    # split into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    
          

if __name__ == "__main__":
    main()
