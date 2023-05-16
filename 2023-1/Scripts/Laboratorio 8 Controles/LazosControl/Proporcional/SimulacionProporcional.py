# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:32:48 2023

@author: Esteban VC

Control Proporcional - TCLAB

Función de Transferencia Continua:
                1.80909 exp(-11.8264s)
    G(s) =  ----------------------
                205.9033s + 1
                
Función de Transferencia Discreta:
                 0.2895 z + 0.08482
G(z) = z^(-1) * ---------------------   Ts = 47
                    z - 0.7931
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import tclab_cae.tclab_cae as tclab
import time
from datetime import date

#Importar funciones modelo NO Lineal
import Funtions as labtool 

#Función de retardo
def delay_time(sleep_max, prev_time):
    sleep = sleep_max - (time.time() - prev_time)
    if sleep >= 0.01:
        time.sleep(sleep - 0.01)
    else:
        time.sleep(0.01)
        
    # Record time and change in time
    t = time.time()
    return t    

plt.close()
# Función de Transferencia Discreta
Ts   =  47                  #Periodo de Muestreo
numz =  np.array([0.2895, 0.08482])   #Numerador
denz =  np.array([1, -0.7931])        #Denominador
d    =  1                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
# print(Gz)

#===========================================================#
# Connect to Arduino
lab = tclab.TCLab_CAE()
start_time = time.time()
prev_time = start_time
#sleep time
sleep_max = 1.0
#===========================================================#

# Parametros del Modelo No Lineal
Ta = lab.T2  
Tinit = lab.T3

#Crea los vectores
tsim = 1000                #Tiempo de simulacion (segundos)
nit = int(tsim/1)          #Numero de Iteraciones
t = np.zeros(nit)          #Tiempo

#Vectores del proceso Real
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
e = np.zeros(nit)           #Vector de error

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[Ts*3:] = 40

# Control Proporcional
Kc = 1.8
bias = 0
kss = dcgain(Gz)  #Ganancia del sistema

#Crear plot
plt.figure(figsize=(10,7))
plt.ion() #Enable interactive mode
plt.show()

try:
    #Lazo Cerrado de Control
    for k in range(nit):
        
        #calculo del tiempo de graficación
        tm = delay_time(sleep_max, prev_time)
        prev_time = tm
        t[k] = np.round(tm - start_time) - 1
        
        #=====================================================#
        #============         PROCESO REAL        ============#
        #=====================================================#
        y[k] = lab.T3 
        
        #=====================================================#
        #============       CALCULAR EL ERROR     ============#
        #=====================================================#
        e[k] = r[k] - y[k]
        
        #=====================================================#
        #===========   CALCULAR LA LEY DE CONTROL  ===========#
        #=====================================================#
        bias = (y[k] - y[0]) / kss
        
        if  t[k]%Ts == 0:
            u[k] = Kc * e[k] + bias #Ley de Control
        else:
            u[k] = u[k-1]
            
        #Saturación
        if u[k] > 100:
            u[k] = 100
        elif u[k] <0:
            u[k] = 0
        
        # write Heater (0 -100)
        lab.Q2(u[k])
        
        #Graficar
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(t[0:k],r[0:k],'--k',linewidth=3)
        plt.plot(t[0:k],y[0:k],'r-',linewidth=3)
        plt.legend(['Setpoint', 'Output'])
        plt.ylabel('Temperature (C)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.title('Proportional Control',fontsize=24)

        plt.subplot(2,1,2)
        plt.step(t[0:k],u[0:k],'b-',linewidth=3)
        plt.legend(['Heater'])
        plt.ylabel('Power (%)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.draw()
        plt.pause(0.05)
            
    # Turn off heaters
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    
# Allow user to end loop with Ctrl-C          
except KeyboardInterrupt:
    print('Operación interrumpida por teclado')
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    lab.close()
    plt.savefig('CP_response.png') 
    np.save_txt(t[0:k], y[0:k], u[0:k])    
    
finally:
    # Disconnect from Arduino
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    lab.close()
    print('Shutting down')
    today = date.today()
    formatted_date = today.strftime("%Y_%m_%d")
    name = "ControlProporcional_TcLab" + formatted_date
    labtool.save_txt(t, u, y, r,"SimulacionProporcionalTcLab")    
    plt.savefig(name+".png")
