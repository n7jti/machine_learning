#!/usr/bin/python


import argparse
import numpy as np
from ftplib import FTP

def loadData (f):
  # Load a csv of floats:
  d = np.loadtxt(f, skip_header=1)
  return d

def main ():
  parser = argparse.ArgumentParser(description='Kernalized Perceptron!')
  parser.add_argument('-f', help='input file', default='2013')
  args = parser.parse_args()

  d = loadData(args.f)
  print d


if __name__ == "__main__":
    main()
