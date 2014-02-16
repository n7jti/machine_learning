#!/usr/bin/python

import argparse
import numpy as np
from sklearn import cross_validation
from sklearn import svm

def loadData (subdir):
  # Load a csv of floats:
  data = np.genfromtxt(subdir + '/labels.csv', delimiter=",", skip_header=0)
  
  return data

def main ():
  parser = argparse.ArgumentParser(description='convert from raw features to fivefeatures')
  parser.add_argument('-source', help='subdirectory for the source data')
  args = parser.parse_args()

  y = loadData(args.source);
  dest = open(args.source + '/fivelabels.csv','w')

  for idx in range(y.shape[0]-5):
    dest.write(str(y[idx][1]))
    for j in range(1,5):
      dest.write(','+str(y[idx + j][1]))
    dest.write('\n')


if __name__ == "__main__":
    main()
