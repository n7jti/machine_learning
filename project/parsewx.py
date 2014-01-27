#!/usr/bin/python

import argparse
import re
import gzip

regex = "(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+)\s+"                      + \
        "(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+"            + \
        "(\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+)\s+"                 + \
        "(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\*?\s+(\d+\.\d+)(\*?)\s+(\d+\.\d+)"     + \
        "([ABCDEFGHI])\s+(\d+\.\d+)\s+([01])([01])([01])([01])([01])([01])$"
        
def main ():
  #parser = argparse.ArgumentParser(description='Kernalized Perceptron!')
  #parser.add_argument('-f', help='input file', default='2013')
  #args = parser.parse_args()

  f = gzip.open('727930-24233-2000.op.gz', 'rb')
  file_content = f.read()
  lines = file_content.split('\n')

  prog = re.compile(regex)

  f = True
  for line in lines: 
    if f:
      f=False
      continue
    if len(line) == 0:
      continue
    print line
    result = prog.match(line)
    print result.groups()




if __name__ == "__main__":
    main()

