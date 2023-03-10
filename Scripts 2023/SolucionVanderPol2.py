# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:57:09 2023

@author: Esteban VC
"""

from scipy.integrate import solve_ivp
import numpy as np 
import matplotlib.pyplot as plt  
 
#%%  Programar la EDO 
def Vanderpol(t, x, u, mu): 
     
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

#%%  Condición inicial 

rigido = True
x0 = [2, 0] 
u = 0  

if rigido: 
    t_span = [0, 3000]
    mu = 1000
    metodo = 'Radau'
else:
    t_span = [0, 30]
    mu = 1
    metodo = 'RK45'

#Solucionar la EDO 
sol = solve_ivp(Vanderpol, t_span, x0, method=metodo, args=(u, mu)) 

y = sol.y
t = sol.t

#Graficar 
plt.plot(t, y[:][0])
plt.xlabel("tiempo (s)")
plt.ylabel("Posición - x")
#plt.show() 