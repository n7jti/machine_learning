#!/usr/bin/python2

from scipy import *
import scipy.sparse as sp
import numpy as np

#Entropic quantities
def h2(p):
    lh = 0.0
    if p > 0:
        lh = p * log2(p)
    rh = 0.0
    if p < 1:
        rh = (1 - p) * log2(1 - p)
    return -(lh + rh)
#
def dense_stump_fit(f, y, prob, pplus):
    perm = argsort(f)
    fp = f[perm]; yp = (0.5 * (y + 1))[perm]; pp = prob[perm]
    pl = 0.;  plt = 0.
    pr = 1.0; prt = pplus

    Hc = []
    fprev = fp[0] - 1.0
    for i in range(len(f) - 1):
        if fp[i] != fprev:
            plt += yp[i] * pp[i]; prt -= yp[i] * pp[i]
            pl += pp[i]; pr -= pp[i]
            Hc.append([i, pl * h2(plt/pl) + pr * h2(prt/pr)])
    Hc = array(Hc)
    dc = int(Hc[Hc[:, 1].argmin(), 0])
    fv= fp[dc:dc+2].mean()
    err = ((stump_predict(f, fv) * y < 0) * prob).sum()
    return (fv, err)

def sparse_stump_fit(f, y, prob, pplus):
    #Will only work with CSC column matrices
    #Results are identical with dense stump-fit, if there exists atleast
    #one data point with f_i = 0. If not, the result is no worse. We also
    #take advantage of the convexity of H, by stopping the search when 
    #Hc < Hthresh.
    if not (sp.isspmatrix_csc(f) and f.shape[1] == 1):
        raise ValueError

    nr = f.shape[0]; nz = f.nnz
    #
    sarg = argsort(f.data)
    sdat = f.data[sarg]
    zidx = sdat.searchsorted(0.0)
    sidx = f.indices[sarg]
    #
    Hc = []; Hcur = 1.0; Hthresh = 1e-6

    pl = 0.0; plt = 0.0
    pr = 1.0; prt = pplus
    for i in range(zidx):
        pele = prob[sidx[i]]; cele = 0.5 * (y[sidx[i]] + 1)
        pl += pele; plt += cele * pele
        pr -= pele; prt -= cele * pele
        if i != zidx-1:
            dv = mean(sdat[i:i+2])
        else:
            dv = sdat[i]/2.0
        Hcur = pl * h2(plt/pl) + pr * h2(prt/pr)
        Hc.append([dv, Hcur])
        if Hcur < Hthresh:
            break

    if Hcur > Hthresh:
        pl = 1.0; plt = pplus
        pr = 0.0; prt = 0.0
        for i in range(nz - 1, zidx - 1, -1):
            pele = prob[sidx[i]]; cele = 0.5 * (y[sidx[i]] + 1)
            pl -= pele; plt -= cele * pele
            pr += pele; prt += cele * pele
            if i != zidx:
                dv = mean(sdat[i-1:i+1])
            else:
                dv = sdat[i]/2.0
            Hcur = pl * h2(plt/pl) + pr * h2(prt/pr)
            Hc.append([dv, Hcur])
            if Hcur < Hthresh:
                break

    Hc = array(Hc)
    (dv, Hm) = Hc[Hc[:, 1].argmin(), :]
    didx = sdat.searchsorted(dv)
    if dv >= 0:
        err = pplus - (prob[sidx[didx:]] * (y[sidx[didx:]] > 0)).sum() + (prob[sidx[didx:]] * (y[sidx[didx:]] < 0)).sum()
    else:
        err = 1 - pplus + (prob[sidx[:didx]] * (y[sidx[:didx]] > 0)).sum() - (prob[sidx[:didx]] * (y[sidx[:didx]] < 0)).sum()

    return (dv, err)

#
def stump_fit(f, y, prob, pplus):
    if sp.issparse(f):
        ret = sparse_stump_fit(f, y, prob, pplus)
    else:
        ret = dense_stump_fit(f, y, prob, pplus)
    return ret

def stump_predict(f, fv):
    if sp.issparse(f):
        tmp = array(f.todense())[:, 0]
    else:
        tmp = f
    return 2 * (tmp > fv) - 1

#Tests
def two_clusters(n, a=4.0):
    x = append(randn(n) - a, randn(n) + a)    
    y = zeros(2 * n)
    y[:n] = -1
    y[n:] = 1

    pn = arange(2 * n, dtype=int32)
    np.random.shuffle(pn)
    x = x[pn]
    y = y[pn]

    return (x, y)
