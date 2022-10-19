# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt 

#Programar la EDO
def vanderpol(x, t, u, mu):
    
    # Renombrando los estados
    x1 = x[0]
    x2 = x[1]
    
    #Renombrar las entradas
    
    #Ecuaciones Diferenciales
    dx1dt = x2
    dx2dt = mu*(1-x1**2)*x2 - x1
    
    #Salida
    dx = [dx1dt, dx2dt]
    return dx

#Condición inicial
x0 = [2, 0]
mu = 1000
u = 0 

#Tiempo de integración
t = np.linspace(0, 3000)

#Solucionar la EDO
x = odeint(vanderpol, x0, t, args=(u, mu), rtol=1e-5)

#Graficar
plt.plot(t, x[:,0])
plt.xlabel("tiempo (s)")
plt.ylabel("Posición - x")
plt.show()


















