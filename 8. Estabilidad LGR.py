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

nums = 1
dens = np.convolve([1, 2], [1, 10, 7])

Gs = tf(nums, dens)
print(Gs)

# Sistema Discreto (Dominio de Z)

numz = [0.1021, 0.2566, 0.03881]
denz = [1, -1.216, 1.146, -0.1353]
Ts = 1
Gz = tf(numz, denz, Ts)
print(Gz)

#%% Mapa de Polos y Ceros
"""
plt.figure(1)
pzmap(Gs, grid = True)

plt.figure(2)
pzmap(Gz, grid = True)
"""
# Primera forma de verificar estabilidad mediante la visualización de los polos y ceros del sistema
#%% Respuesta Dinámica
"""
y, t = step(Gs)
plt.figure(3)
plt.plot(t, y)

y, t = step(Gz)
plt.figure(4)
plt.plot(t, y)
"""
# Segunda forma de verificar la estabilidad del sistema, metiendole un escalón a la planta 
# y viendo como responde, es estable si converge a un valor

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




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    