# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:20:15 2022

@author: Esteban

# Rlocus
"""

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *

#G = tf(1, np.convolve([1, 2],[1, -1]))
G = tf([1.84], [115.6, 1])
plt.figure(1)
rlocus(G)

# Lazo Cerrado

retardo = 25.16440193144541
numRetardo, denRetardo = pade(retardo, 1)
G_Retardo = tf(numRetardo, denRetardo)
H = feedback(G*G_Retardo, 1)
y, t = step(H)

plt.figure(2)
pzmap(H)
plt.grid()

plt.figure(3)
plt.plot(t, y)
plt.grid()
