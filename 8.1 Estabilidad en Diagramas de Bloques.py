# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:57:59 2022

@author: Esteban
Estabilidad Diagramas de Bloques
"""

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *

G = tf([1, 1], [1, 6, 4])
k = 500

plt.figure(1)
y, t = step(G)
plt.plot(t, y)

plt.figure(2)
pzmap(G)

# Bloques en serie
kG = series(k, G)
H = feedback(kG, 1)

plt.figure(3)
y, t = step(H)
plt.plot(t, y)

plt.figure(4)
pzmap(H)

"""Al cerrar el lazo, hicimos que se movieran los polos al punto de aumentar tanto la ganancia
que un cero y un polo se anularon por simetría, por otro lado, el polo restante se va a 
menos infinito haciendo que el sistema responde mucho mas rápido pero solo es posible en simulación
ya que tendriamos que meterle energia infinita al sistema para que pueda responder de esta manera
en la vida real."""

