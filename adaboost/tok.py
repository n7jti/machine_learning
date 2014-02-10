#!/usr/bin/python2

from scipy import *
import scipy.sparse as sp
import scipy.linalg as la
#See http://scikit-learn.org/stable/modules/feature_extraction.html
import sklearn.feature_extraction as fe
import os
import re
import string

def parse_mail(txt):
    txtl = txt.splitlines()
    retl = list(txtl)
    for i in range(len(txtl)):
        if re.match("^In article(.*?)writes:$", txtl[i]) \
           or re.match("^.?.?\>.", txtl[i]) \
           or re.match("^.?.?\:.", txtl[i]) :
            retl[i] = ""
    return filter(lambda x : x in string.printable, "\n".join(filter(lambda x: x != "", retl)))

def fill_corpus(groupnames, dirname = "20_newsgroups/"):
    corpus = []
    for gname in groupnames:
        ret = []
        for fname in os.listdir(dirname + gname):
            ret.append(parse_mail(open(dirname + gname + "/" + fname).read().split("\n\n", 1)[1]).replace(chr(0xff), ""))
        corpus.append(ret)
    return corpus
