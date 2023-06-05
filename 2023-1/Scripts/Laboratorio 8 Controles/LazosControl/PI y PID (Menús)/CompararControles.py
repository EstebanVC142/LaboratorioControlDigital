# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:59:43 2023

@author: Esteban VC
"""

    # -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:41:11 2023

@author: Esteban VC
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close()
# Cargar datos de la simulación
data_sim1 = np.loadtxt('Escogio el metodo PI Control Pole Assignment.txt',delimiter=',',skiprows=1)
t_simZN = data_sim1[:,0].T 
u_simZN = data_sim1[:,1].T
y_simZN = data_sim1[:,2].T
r_simZN = data_sim1[:,3].T

data_sim2 = np.loadtxt('Real PI Control Pole Assignment.txt',delimiter=',',skiprows=1)
t_simIAE = data_sim2[:,0].T 
u_simIAE = data_sim2[:,1].T
y_simIAE = data_sim2[:,2].T
r_simIAE = data_sim2[:,3].T

# data_sim3 = np.loadtxt('Escogio el metodo PID Control IAET Roveri.txt',delimiter=',',skiprows=1)
# t_simITAE = data_sim3[:,0].T 
# u_simITAE = data_sim3[:,1].T
# y_simITAE = data_sim3[:,2].T
# r_simITAE = data_sim3[:,3].T

# data_sim4 = np.loadtxt('Escogio el metodo PID Control Cohen-Coon.txt',delimiter=',',skiprows=1)
# t_simCC = data_sim4[:,0].T 
# u_simCC = data_sim4[:,1].T
# y_simCC = data_sim4[:,2].T
# r_simCC = data_sim4[:,3].T

#Setpoint
r = np.zeros(len(u_simZN))
r[:] = y_simZN[0]
r[2*47:] = 40

# Gráfica de la temperatura (setpoint y salida)
plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)

# Simulación
plt.plot(t_simZN, y_simZN, linestyle='-', color='r', linewidth=3, label='PI')
plt.plot(t_simIAE, y_simIAE, linestyle='-', color='b', linewidth=3, label='PID')
plt.plot(t_simIAE, r_simIAE, linestyle='--', color='k', linewidth=3, label='Setpoint')

plt.legend(loc='best')
plt.ylabel('Temperature (C)', fontsize=18)
plt.xlabel('Time(s)', fontsize=18)
plt.title('PI Control Pole Assignment', fontsize=24)

# Gráfica de la potencia del calentador
plt.subplot(2, 1, 2)

# Simulación
plt.step(t_simZN, u_simZN, linestyle='-', color='r', linewidth=3, label='P')
plt.step(t_simIAE, u_simIAE, linestyle='-', color='b', linewidth=3, label='PI')
# plt.step(t_simCC, u_simCC, linestyle='-', color='g', linewidth=3, label='PID')

plt.legend(loc='best')
plt.ylabel('Power (%)', fontsize=18)
plt.xlabel('Time(s)', fontsize=18)
plt.show()
plt.savefig('CompararacionPIAP.png') 
