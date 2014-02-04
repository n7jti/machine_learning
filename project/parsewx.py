#!/usr/bin/python

import argparse
import re
import gzip

from os import listdir
from os.path import isfile, join

regex = "(?P<station>\d+)\s+(?P<wban>\d+)\s+" + \
        "(?P<date>\d+)\s+" + \
        "(?P<temp>-?\d+\.\d+)\s+(?P<temp_count>\d+)\s+" + \
        "(?P<dew_point>-?\d+\.\d+)\s+(?P<dew_point_count>\d+)\s+" + \
        "(?P<sea_level_pressure>\d+\.\d+)\s+(?P<sea_level_pressure_count>\d+)\s+" + \
        "(?P<station_pressure>\d+\.\d+)\s+(?P<station_pressure_count>\d+)\s+" + \
        "(?P<visibility>\d+\.\d+)\s+(?P<visibility_count>\d+)\s+" + \
        "(?P<wind_speed>\d+\.\d+)\s+(?P<wind_speed_count>\d+)\s+" + \
        "(?P<max_wind>\d+\.\d+)\s+" + \
        "(?P<gust>\d+\.\d+)\s+" + \
        "(?P<high>-?\d+\.\d+)(?P<high_indicator>\*?)\s+" + \
        "(?P<low>-?\d+\.\d+)(?P<low_indicator>\*?)\s+" + \
        "(?P<precip>\d+\.\d+)(?P<precip_flag>[ABCDEFGHI])?\s+" + \
        "(?P<snow_depth>\d+\.\d+)\s+" + \
        "(?P<fog>[01])(?P<rain>[01])(?P<snow>[01])(?P<hail>[01])(?P<thunder>[01])(?P<tornado>[01])$"

def features (dest):
  f = open(dest + '/features.txt','w')
  f.write('temp\n')
  f.write('dew_point\n')
  f.write('sea_level_pressure\n')
  f.write('visibility\n')
  f.write('wind_speed\n')
  f.write('max_winds\n')
  f.write('gust\n')
  f.write('high\n')
  f.write('low\n')
  f.close()

def labels (dest):
  f = open(dest + '/labels.txt', 'w')
  f.write('fog\n')
  f.write('rain\n')
  f.write('hail\n')
  f.write('snow\n')
  f.write('thunder\n')
  f.write('tornadn')
  f.close()

def writeData (result, data):
  data.write(result.group('temp'))
  data.write(",")

  data.write(result.group('dew_point'))
  data.write(",")

  data.write(result.group('sea_level_pressure'))
  data.write(",")

  data.write(result.group('visibility'))
  data.write(",")

  data.write(result.group('wind_speed'))
  data.write(",")

  # treat no max winds reported as winds zero
  maxWind = result.group('max_wind')
  if maxWind == '999.9':
    maxWind = '0.0'
  data.write(maxWind)
  data.write(",")

  # treat no gust reported as zero gusts
  gust = result.group('gust')
  if gust == '999.9':
    gust = '0.0'
  data.write(gust)
  data.write(",")

  data.write(result.group('high'))
  data.write(",")

  data.write(result.group('low'))
  data.write('\n')

def fixupLabel (label):
  if label == '0':
    label = '-1'
  return label

def writeLabel (result, labels):

  fog = fixupLabel(result.group('fog'))
  labels.write(fog)
  labels.write(',')

  rain = fixupLabel(result.group('rain'))
  labels.write(rain)
  labels.write(',')

  snow = fixupLabel(result.group('snow'))
  labels.write(snow)
  labels.write(',')

  hail = fixupLabel(result.group('hail'))
  labels.write(hail)
  labels.write(',')

  thunder = fixupLabel(result.group('thunder'))
  labels.write(thunder)
  labels.write(',')

  tornado = fixupLabel(result.group('tornado'))
  labels.write(tornado)
  labels.write('\n')

def parseFile (file, data, label):
  data_content = file.read()
  lines = data_content.split('\n')

  prog = re.compile(regex)

  f = True
  for line in lines: 
    if f:
      f=False
      continue
    if len(line) == 0:
      continue

    result = prog.match(line)
    writeData(result,data)
    writeLabel(result,label)
        
def main ():
  parser = argparse.ArgumentParser(description='Parse the raw weather data into CSV files')
  parser.add_argument('-source', help='subdirectory for the source data')
  parser.add_argument('-dest', help='subdirectory for the destination data')
  args = parser.parse_args()

  data = open(args.dest + '/data.csv','w')
  label = open(args.dest + '/labels.csv', 'w')
  features(args.dest)
  labels(args.dest)

  mypath = args.source
  fileList = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
  fileList.sort()

  for fileName in fileList:
    print "parsing", fileName
    file = gzip.open(join(mypath,fileName), 'rb')
    parseFile(file,data,label)

  data.close()
  label.close()

if __name__ == "__main__":
    main()

