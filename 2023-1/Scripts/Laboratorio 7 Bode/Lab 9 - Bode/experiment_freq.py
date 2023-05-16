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

# La entrada de estimulo siempre es senoidal
#Frecuencia (Variar la Frecuencia)
ws = 10**2 # ws= 2*pi / Ts
#amplitud
A = 1
#fase
alpha = 0

# Vector Tiempo
t = np.arange(0,50/ws,0.1/ws)

#entrada 
u = A * np.sin(ws*t + alpha)

#Salida de la planta
y, tout, xout = lsim(G, u, t)


plt.plot(t, u, '-r', t, y, '-b')
plt.plot((t[0],t[-1]),(0,0),'-k')
plt.plot((t[0],t[-1]),(A,A),'-k')
plt.plot((t[0],t[-1]),(-A,-A),'-k')
plt.title('Respuesta en Frecuencia', fontsize=16)
plt.ylabel('Amplitud', fontsize=14)
plt.xlabel('Tiempo (s)', fontsize=14)
plt.legend(['Entrada', 'Salida'])




