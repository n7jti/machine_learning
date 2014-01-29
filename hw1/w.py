#!/usr/bin/python

import numpy as np

def loadData ():

  #
  #  upvote
  #

  # Load a text file of floats:
  wData = np.loadtxt("YelpW.txt", dtype=np.float)

  # Load a text file of strings:
  uvFeat = open("data/upvote_features.txt").read().splitlines()



  return wData, uvFeat

def main ():
  wData, uvFeat = loadData()

  print "i, w, label"
  for i in range(wData.shape[0]):
    if wData[i] > 0.001:
      print i, ",", wData[i], ",", uvFeat[i]
      
  

if __name__ == "__main__":
    main()

