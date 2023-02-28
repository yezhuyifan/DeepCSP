#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
To post-process the generated fake samples:
(1) Round and divide the parameters.
'''
import numpy as np
import pandas as pd

from bisect import bisect_left


def find_spacegroup(x):
    l = list(range(1, 12))
    if (x >= l[-1]):
        return l[-1]
    elif x <= l[0]:
        return l[0]
    pos = bisect_left(l, x)
    before = l[pos - 1]
    after = l[pos]
    if after - x < x - before:
       return after
    else:
       return before

def find_z(x):
    l = [1, 2, 4, 8, 16]
    if (x >= l[-1]):
        return l[-1]
    elif x <= l[0]:
        return l[0]
    pos = bisect_left(l, x)
    before = l[pos - 1]
    after = l[pos]
    if after - x < x - before:
       return after
    else:
       return before

def find_alpha_gamma(x):
    l = list(range(60, 121, 1))
    if (x >= l[-1]):
        return l[-1]
    elif x <= l[0]:
        return l[0]
    pos = bisect_left(l, x)
    before = l[pos - 1]
    after = l[pos]
    if after - x < x - before:
       return after
    else:
       return before

def find_beta(x):
    l = list(range(60, 121, 1))
    if (x >= l[-1]):
        return l[-1]
    elif x <= l[0]:
        return l[0]
    pos = bisect_left(l, x)
    before = l[pos - 1]
    after = l[pos]
    if after - x < x - before:
       return after
    else:
       return before
