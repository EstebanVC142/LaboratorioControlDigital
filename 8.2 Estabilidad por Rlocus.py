# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:20:15 2022

@author: Esteban

# Rlocus
"""

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *

G = tf(1, np.convolve([1, 2],[1, -1]))
plt.figure(1)
rlocus(G)

# Lazo Cerrado

k = 200
kG = series(k, G)
H = feedback(kG, 1)
y, t = step(H)
plt.figure(2)
plt.plot(t, y)
plt.figure(3)
pzmap(H)
