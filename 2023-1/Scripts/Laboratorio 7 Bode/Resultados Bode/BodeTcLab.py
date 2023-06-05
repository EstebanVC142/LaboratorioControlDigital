# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:32:09 2022

@author: Sergio
"""
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from controlcae import *

plt.close()
# G = tf(0.33427372912516046, [24.001052490258118, 1]) # Modelo obtenido por optimización
# G = tf(1.84, [115.6, 1]) # Modelo por curva de reacción
G = tf([0.2428, 0.3552], [1, 0.7938, 0], dt = 26.7) # Modelo de curva de reaccion discretizado 
k = 1
cG = series(k, G)
bode(cG)

# Analisis de Estabilidad
gm, pm, Wcg, Wcp = margin(cG)

print(G)
# gm, pm, Wcg, Wcp = margin_plot(cG)
print('Margen de Ganancia:', 20*np.log10(gm))
print('Margen de Fase:', 20*np.log10(pm))
print('Frecuencia de Cruce de Fase:', Wcg)
print('Frecuencia de Cruce de Ganancia:', Wcp)