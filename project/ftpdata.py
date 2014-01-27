#!/usr/bin/python

from ftplib import FTP
import argparse

def main ():
  parser = argparse.ArgumentParser(description='Kernalized Perceptron!')
  parser.add_argument('-b', type=int, help='the first year to get', default='2013')
  parser.add_argument('-e', type=int, help='the last year to get', default='2013')
  parser.add_argument('-s', help='the sation to get', default='727930-24233')
  args = parser.parse_args()

  #ftp://ftp.ncdc.noaa.gov/pub/data/gsod/2013/727930-24233-2013.op.gz 

  ftp = FTP('ftp.ncdc.noaa.gov')
  ftp.login();
  ftp.cwd('pub/data/gsod') 
  
  start = args.b
  end = args.e

  for year in range(start, end):
    filename = args.s + '-' + str(year) +'.op.gz'
    ftp.cwd(str(year))
    ftp.retrbinary('RETR ' +filename, open(filename, 'wb').write)
    ftp.cwd('..')

  ftp.quit()


if __name__ == "__main__":
    main()
