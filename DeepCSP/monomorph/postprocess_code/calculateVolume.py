#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
To post-process the generated samples:
(2) Calculate the cell volume.
'''
from math import sin, cos, radians, sqrt
import numpy as np
import pandas as pd


def triclinic(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = radians(alpha), radians(beta), radians(gamma)
    if (1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)) < 0:
        return 0
    v = a*b*c * sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)) # Triclinic
    return v

def monoclinic(a, b, c, beta):
    beta = radians(beta)
    v = a*b*c * sin(beta) # Monoclinic
    return v

def orthorhombic(a, b, c):
    v = a*b*c # Orthorhombic
    return v
