# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:26:14 2020

@author: vedenev
"""

import numpy as np

a = np.arange(5, dtype=np.float32) + 0.1
a.tofile('a.bin')