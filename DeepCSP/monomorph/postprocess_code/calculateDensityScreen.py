#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
To post-process the generated samples:
(3) Calculate the cell density.
'''
from scipy.constants import Avogadro
import numpy as np
import pandas as pd


def density(z, mw, na, v):
    den = ((z*mw)/(na*v))*(10**24)
    return den
