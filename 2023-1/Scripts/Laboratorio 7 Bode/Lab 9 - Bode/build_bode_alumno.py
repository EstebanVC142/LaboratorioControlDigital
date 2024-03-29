# -*- coding: utf-8 -*-
"""
Diagrama de Bode
¿Cómo se hace un diagrama de Bode?

@author: Sergio
"""

import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *

plt.close()

# Creamos una función de transferencia
# que representa mi sistema o mi máquina real
G = tf(1, [1, 1])
print(G)

# Datos Experimentales
w   = [10**-2, 10**-1, 10**0, 10**1, 10**2]
mag = [1, 0.99375, 0.7063, 0.1025, 0.01664]
# fase = Delta tiempo * ws 
pha = [0, -0.09962, -0.7849,  -1.452, -1.519]

magdB = list(map(lambda i: 20*np.log10(i), mag))
phad = list(map(lambda i: np.rad2deg(i), pha))

plt.figure(1)
bode(G)
plt.figure(1)
plt.subplot(211)
plt.semilogx(w, magdB)
plt.grid()
plt.ylabel('Magnitud (dB)')
plt.xlabel('Frecuencia (rad/s)')
plt.subplot(212)
plt.semilogx(w, phad)
plt.grid()
plt.ylabel('Fase (deg)')
plt.xlabel('Frecuencia (rad/s)')