#!/usr/bin/python2

from scipy import *
import scipy.sparse as sp
import numpy as np

#Tests
def two_lines(n):
    a=4.0

    theta = pi/4; c = cos(theta); s = sin(theta)
    R = array([[c, -s],
               [s, c]])
    d = diag(array([10, 0.1]))

    m = 2.5
    xm = d.dot(R.T.dot(randn(2, n))) + array([[0], [m]])
    xm = R.dot(xm)

    xp = d.dot(R.T.dot(randn(2, n))) + array([[0], [-m]])
    xp = R.dot(xp)

    x = append(xm, xp, 1)
    x = x.T

    y = zeros(2 * n)
    y[:n] = 1; y[n:] = -1

    prm = arange(2 * n, dtype=int32)
    np.random.shuffle(prm)
    y = y[prm]
    x = x[prm, :]

    return (x, y)

def four_clusters(n):
    a = 4.0
    xm = append(randn(n, 2) + array([a, a]), randn(n, 2) + array([-a, -a]), 0)
    xp = append(randn(n, 2) + array([-a, a]), randn(n, 2) + array([a, -a]), 0)

    x = append(xm, xp, 0)
    y = zeros(4 * n)
    y[:2 * n] = 1
    y[2 * n:] = -1

    pn = arange(4 * n, dtype=int32)
    np.random.shuffle(pn)
    x = x[pn, :]
    y = y[pn]

    return (x, y)
