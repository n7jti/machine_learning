#!/usr/bin/python2
from scipy import *
import scipy.sparse as sp
import dstump as ds

(f, y) = ds.two_clusters(100)
pr = ones(len(y))/len(y)

#This quantity is invariant for each Adaboost step, and helps us take 
#advantage of sparsity.
pplus = sum(pr * (y > 0))

#The decision stump training routine accepts either a dense 1-d 
#array or a sparse 1-d CSC matrix. The resulting decision variable 
#might be different for dense and sparse data, but the errors are
#the same. See implementation for details.
(dv, err) = ds.stump_fit(f, y, pr, pplus)

#Inplace transpose of a CSR matrix gives a CSC matrix.
fs = sp.csr_matrix(f).T 
(dvs, errs) = ds.stump_fit(fs, y, pr, pplus)
