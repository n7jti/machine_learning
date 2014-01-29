#!/bin/bash


python 32b.py 50 75 5 1 > 50-75.csv &
python 32b.py 50 150 5 1 > 50-150.csv &
python 32b.py 50 5000 5 1 > 50-5000.csv &
python 32b.py 100 75 5 1 > 100-75.csv &
python 32b.py 100 150 5 1 > 100-150.csv &
python 32b.py 100 5000 5 1 > 100-5000.csv &

