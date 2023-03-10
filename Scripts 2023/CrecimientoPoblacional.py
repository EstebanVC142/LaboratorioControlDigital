# -*- coding: utf-8 -*-
"""
Spyder Editor

Equipo: ¤ Juan David Corrales
        ¤ Daniel Montoya Lopez
        ¤ Esteban Vásquez Cano
"""

from scipy.integrate import odeint, solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def CrecimientoPoblacional(t, y, k = 0.01, A = 100):
    
    y1 = y[0]
    y2 = y[1]
    
    dy1dt = y2
    dy2dt = k*y1*(A-y1)
    
    dydt = [dy1dt, dy2dt]
    
    return dydt

# Valores de las constantes
#A = 100 # celulas*mililitro
#k = 0.01 # mililitros/celula

# Condiciones iniciales
t0 = 0.0
y0 = np.array([10.0, 0])

# Intervalo para encontrar la solución
tf = 24.0 # Horas
tiempoSolucion = np.array([t0, tf])
t = np.arange(t0, tf, 0.1)

# Solucionando la ecuación diferencial 
solucion = solve_ivp(CrecimientoPoblacional, tiempoSolucion, y0, t_eval = t, method = 'RK45')

y = solucion.y
t = solucion.t

#Graficar
plt.figure(2)
plt.plot(t, y[:][0], color = 'r', label = "Solución Númerica")
plt.legend(loc = 'best')
plt.xlabel("Tiempo (horas)", fontsize = 14)
plt.ylabel("Densidad de población (celulas/ml)", fontsize = 14)
plt.grid()
plt.show() 