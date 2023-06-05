# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:41:11 2023

@author: Esteban VC
"""

import numpy as np
import matplotlib.pyplot as plt

# Cargar datos de la simulación
data_sim = np.loadtxt('SimulacionProporcional.txt',delimiter=',',skiprows=1)
t_sim = data_sim[:,0].T 
u_sim = data_sim[:,1].T
y_sim = data_sim[:,2].T
r_sim = data_sim[:,3].T

# Cargar datos reales
data_real = np.loadtxt('SimulacionProporcionalTcLab.txt',delimiter=',',skiprows=1)
t_real = data_real[:,0].T 
u_real = data_real[:,1].T
y_real = data_real[:,2].T
r_real = data_real[:,3].T

#Setpoint
r = np.zeros(len(u_real))
r[:] = y_real[0]
r[2*47:] = 40

# Gráfica de la temperatura (setpoint y salida)
plt.figure()
plt.subplot(2, 1, 1)

# Simulación
plt.plot(t_sim, y_sim, linestyle='-', color='r', linewidth=3, label='Simulation')
# Datos reales
plt.plot(t_real, y_real, linestyle='-', color='g', linewidth=3, label='Real')

plt.plot(t_real, r_real, linestyle='--', color='k', linewidth=3, label='Setpoint')
# plt.plot(t_sim, r_sim, linestyle='--', color='k', linewidth=3, label='Setpoint')

plt.legend()
plt.ylabel('Temperature (C)', fontsize=18)
plt.xlabel('Time(s)', fontsize=18)
plt.title('Proportional Control', fontsize=24)

# Gráfica de la potencia del calentador
plt.subplot(2, 1, 2)

# Simulación
plt.step(t_sim, u_sim, linestyle='-', color='b', linewidth=3, label='Simulation')
# Datos reales
plt.step(t_real, u_real, linestyle='-', color='c', linewidth=3, label='Real')

plt.legend(['Heater (simulation)', 'Heater (real)'])
plt.ylabel('Power (%)', fontsize=18)
plt.xlabel('Time(s)', fontsize=18)
plt.show()
plt.savefig('CompararacionProporcional.png') 
