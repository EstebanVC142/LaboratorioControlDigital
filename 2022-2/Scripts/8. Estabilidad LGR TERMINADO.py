# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:25:00 2022

@author: Esteban

Analizar los polos de las funciones de transferencia
"""

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *

# Sistema continuo (Dominio de S)

#nums = 1
#dens = np.convolve([1, 2], [1, 10, 7])

nums = 1.84
dens = [115.6, 1]   

Gs = tf(nums, dens)  # Función de Primer Orden
print(Gs)

# Sistema Discreto (Dominio de Z)
"""
numz = [0.1021, 0.2566, 0.03881]
denz = [1, -1.216, 1.146, -0.1353]
"""

numz = [0.02428, 0.3552]
denz = [1, -0.7938, 0]  
Ts = 26.70049088773397

Gz = tf(numz, denz, Ts) # Función de Primer Orden Discretizada
print(Gz)

#%% Mapa de Polos y Ceros

plt.figure(1)
pzmap(Gs, grid = True)
plt.title('Mapa de Polos y Ceros en tiempo continuo.')

plt.figure(2)
pzmap(Gz, grid = True)
plt.title('Mapa de Polos y Ceros en tiempo discreto.')

# Primera forma de verificar estabilidad mediante la visualización de los polos y ceros del sistema
#%% Respuesta Dinámica

y, t = step(Gs)
plt.figure(3)
plt.plot(t, y)
plt.title('Respuesta al escalón en tiempo continuo.')
plt.grid()

y, t = step(Gz)
plt.figure(4)
plt.plot(t, y)
plt.title('Respuesta al escalón en tiempo discreto.')
plt.grid()

# Segunda forma de verificar la estabilidad del sistema, metiendole un escalón a la planta 
# y viendo como responde, es estable si converge a un valor y en este caso converge, por o que es estable

#%% Analisis del Lugar Geoetrico de las Raices

retardo = 25.16440193144541 # Retardo de la planta para el modelo de primer orden
numRetardo, denRetardo = pade(retardo, 1) # Aproximación del retardo por medio de Pade, agrega un polo y cero al sistema
G_Retardo = tf(numRetardo, denRetardo)
Gs_Retardo = Gs*G_Retardo
H = feedback(Gs_Retardo, 1) # Se cierra el lazo con el retardo integrado
y, t = step(H)

plt.figure(5)
pzmap(H, grid = True)
plt.title('Mapa de Polos y Ceros en tiempo continuo con retardo.')

plt.figure(6)
plt.plot(t, y)
plt.title('Respuesta al escalón en tiempo continuo con retardo.')
plt.grid()

plt.figure(7)
rlocus(H)
plt.title('Lugar Geometrico de las raices.')

#%% Analisis de Polos y Ceros

print('='*40)
print('Polos Continuos: ')
i = 1
for p in pole(Gs):
    print(f'Polo {i} = {p}')
    i += 1
    
print('='*40)
print('Ceros Continuos: ')
i = 1
for p in zero(Gs):
    print(f'Ceros {i} = {p}')
    i += 1
    
print('='*40)
print('Polos Discretos: ')
i = 1
for p in pole(Gz):
    print(f'Polo {i} = {p}')
    i += 1  

print('='*40)    
print('Ceros Discretos: ')
i = 1
for p in zero(Gz):
    print(f'Ceros {i} = {p}')
    i += 1

print('='*40)    
print('Magnitud del Polo Discreto: ')
print(np.abs(pole(Gz)))






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    