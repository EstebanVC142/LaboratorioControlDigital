# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:51:47 2022

@author: Esteban
"""

from control.matlab import *
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def graficar(t,y,u,msg):
    #Modifica el tamaño de los ejes
    plt.rcParams.update({'font.size':14})
    
    plt.subplot(211)
    plt.plot(t, y, '-r', linewidth = 2)
    plt.grid()
    plt.ylabel('Output')
    plt.title(msg)
    plt.subplot(212)
    plt.plot(t, u, '-k', linewidth = 2)
    plt.grid()
    plt.ylabel('input')
    plt.xlabel('time')
    plt.show()

plt.close('all')
# Leyendo el archivo donde estan guardados los datos
data  = np.loadtxt('data.txt', delimiter = ',', skiprows = 1)

# Extrayendo los datos de las columnas 
Ts = 1.00249528884
x0 = 40
t = data[:, 0].T
u = data[:, 1].T
y = data[:, 2].T
t_new = np.linspace(0,595,599)

ucopy = u.copy()
tcopy = t.copy()

graficar(t, y, u, 'TClab')

# %%Recortar los datos
index = np.where(u > 0)
st = index[0][0]
ur = u[st:]
yr = y[st:]
tr = t[st:]

plt.figure()
graficar(tr, yr, ur, 'Datos recortados')

# %% transladar los datos 
ut = ur - ur[0]
yt = yr - yr[0]
tt = tr - tr[0]

plt.figure()
graficar(tt, yr, ut, 'Datos transladados y recortados')

# %% calculando tau y theta 
yt1 = yt[-1] * 0.283
yt2 = yt[-1] * 0.632

A = np.array([[1, 1/3],[1, 1]])
b = np.array([yt1, yt2])
xP = np.linalg.inv(A) @ b 
#print(x) # En xP quedan arrojados los valores de tau y theta

# %% Construyendo el modelo de primer orden con retardo (POR)

K = 1.5
tau = xP[0]
theta = xP[1]

if theta < 0:
    theta = 0
    
G = tf(K, [tau, 1])
tdelay = np.arange(0, theta, Ts)

yg, tg, xout = lsim(G, u, t_new)
yg += x0

# Desplazamientos temporales del retardo
tg = np.append(tdelay, tg + theta)
yg = np.append(np.ones(len(tdelay))*yg[0], yg)

plt.figure()
graficar(t, y, u, 'Planta vs Modelo POR')
plt.subplot(211)
plt.plot(tg, yg, linewidth = 2, label = 'Modelo POR')
plt.legend(loc = 'best')

# %% Calculando los parametros para el modelo de segundo orden (SOR)

t1 = yt[-1] * 0.15
t2 = yt[-1] * 0.45
t3 = yt[-1] * 0.75

X = (t2 - t1)/(t3 - t1)

E = (0.0805 - (5.547 * (0.475 - X)**2))/(X - 0.356)

if 0 < E < 1 :
    F2 = 0.708*(2.811)**E
else:
    F2 = (2.6*E) - 0.6

Wn = F2/(t3 - t1)

F3 = 0.922*(1.66)**E

thetaS = t2 - (F3/Wn)

if thetaS < 0:
    thetaS = 0

if 0 < E < 1:
    GpS = tf([K*(Wn**2)],[1, 2*E*Wn, Wn**2])
    
else: 
    tau1 = ((E) - np.sqrt(E**2 - 1))/(Wn)
    tau2 = ((E) + np.sqrt(E**2 - 1))/(Wn)
    
    GpS = tf(K, np.convolve([tau1, 1],[tau2, 1]))
    
tdelayS = np.arange(0, thetaS, Ts)

ygS, tgS, xout = lsim(GpS, u, t_new)
ygS += x0

# Desplazamientos temporales del retardo
tgS = np.append(tdelayS, tgS + thetaS)
ygS = np.append(np.ones(len(tdelayS))*ygS[0], ygS)

plt.figure()
graficar(t, y, u, 'Planta vs Modelo SOR')
plt.subplot(211)
plt.plot(tgS, ygS, linewidth = 2, label = 'Modelo SOR')
plt.legend(loc = 'best')

plt.figure()
graficar(t, y, u, 'Planta vs Modelos POR y SOR')
plt.subplot(211)
plt.plot(tgS, ygS, linewidth = 2, label = 'Modelo SOR')
plt.legend(loc = 'best')
plt.subplot(212)
plt.plot(tg, yg, linewidth = 2, label = 'Modelo POR')
plt.legend(loc = 'best')




        





