#!/usr/bin/python2

import numpy as np


def main ():
  print "hello world"

  a = np.arange(10)
  b=a*2
  c=b*2
  d=c*2

  out=np.vstack((a,b,c,d))

  print "out", out

  np.savetxt("save.txt", out);

  A = np.loadtxt("save.txt")

  d,c,b,a = np.vsplit(A,4)

  print "a", a
  print "b", b
  print "c", c
  print "d", d


if __name__ == "__main__":
    main()
