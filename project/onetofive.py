#!/usr/bin/python
from collections import deque
import argparse


def main ():
  parser = argparse.ArgumentParser(description='Parse the one file into a five-file')
  parser.add_argument('-source', help='source file')
  parser.add_argument('-dest', help='destination finle')
  args = parser.parse_args()


  source = open(args.source)
  dest = open(args.dest, 'w')
  queue = deque()
  for line in source:
    queue.append(line);
    if len(queue) == 6:
      queue.popleft()

    if len(queue) == 5:
      nextline = line.strip()
      for indx in range(1,5):
        nextline = nextline + ',' + queue[indx].strip()
      dest.write(nextline)
      dest.write('\n')


if __name__ == "__main__":
    main()
