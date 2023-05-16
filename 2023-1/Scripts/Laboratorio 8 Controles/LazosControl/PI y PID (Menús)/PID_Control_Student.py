# -*- coding: utf-8 -*-
"""
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
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import sys
from menus import *
#Importar funciones modelo NO Lineal (Elaborados por el grupo)
import Funtions as labtool  
plt.close()

Ts = 47                 #Periodo de Muestreo
#Función de transferencia continua
K = 1.80909
theta1 = 11.8264
tau = 205.9033
#Corrección del retardo para control discreto
theta = theta1 + Ts/2

# Función de Transferencia Discreta
numz =  np.array([0.2895, 0.08482])   #Numerador
denz =  np.array([1, -0.7931])        #Denominador
d    =  1                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
# print(Gz)

# Parametros del Modelo No Lineal
Tinit = ri= 28
Ta=[Tinit]

#Crea los vectores
tsim = 1800                #Tiempo de simulacion (segundos)
nit = int(tsim/Ts)          #Numero de Iteraciones
t = np.arange(0, (nit)*Ts, Ts)  #Tiempo
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
y[:] = Tinit                #Inicializa el vector con Tinit
e = np.zeros(nit)           #Vector de error
q = np.zeros(nit)           #Vector de disturbio
q[25:] = 6

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[5:] = 40

Des = f'Escogio el metodo P'
_control = f'controlled by: P'

if metodo == 1: #Ziegler-Nichols
    if control == 1: #P
        kp=tau/(K*theta)
        ti=np.infty
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt = ('Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    elif control == 2: #PI
        kp=(0.9*tau)/(K*theta)*0.5
        ti=theta*3.3
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt = ('I Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    else: #PID
        kp=(1.2*tau)/(K*theta)
        ti=2*theta
        td=0.5*theta
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('ID Ziegler-Nichols')
        print(f'{Des}{namecrlt}')
        
if metodo == 2: #IAE Rovira
    if control == 1: #PI
        kp=(1/K)*((0.984)*((theta/tau)**-0.986))
        ti=tau/(0.608*((theta/tau)**-0.707))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('IAE Rovira')
        print(f'{Des}{namecrlt}')
    else: #PID Rovira
        kp=(1/K)*((1.435)*((theta/tau)**-0.921))
        ti=tau/(0.878*((theta/tau)**-0.749))
        td=tau*(0.482*((tau/theta)**1.137))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('D IAE Rovira')
        print(f'{Des}{namecrlt}')
        
if metodo == 3: #IAET Roveri
    if control == 1: #PI
        kp=(1/K)*((0.859)*((theta/tau)**-0.977))
        ti=tau/(0.674*((theta/tau)**-0.68))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('IAET Roveri')
        print(f'{Des}{namecrlt}')
    else: #PID
        kp=(1/K)*((1.357)*((theta/tau)**-0.947))
        ti=tau/(0.842*((theta/tau)**-0.738))
        td=tau*(0.381*((tau/theta)**0.995))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('D IAET Roveri')
        print(f'{Des}{namecrlt}')
        
if metodo == 4: #COHEN-COON
    if control == 1: #P
        kp=(1.03+(0.35*(theta/tau)))*(tau/(K*theta))
        ti=0
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt = ('COHEN-COON')
        print(f'{Des} {namecrlt}')
    if control == 2: 
        kp=(0.9+(0.083*(theta/tau)))*(tau/K*theta)
        ti=(theta*(0.9+(0.083*theta/tau)))/(1.27+(0.6*(theta/tau)))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('I COHEN-COON')
        print(f'{Des}{namecrlt}')
    else: #PID
        kp=(1.35+(0.25*theta/tau))*(tau/K*theta)
        ti=(theta*(1.35+(0.25*theta/tau)))/(0.54+(0.33*theta/tau))
        td=(0.5*theta/(1.35+(0.25*theta/tau)))
        incontrolabilidad = theta / tau        
        print(incontrolabilidad)
        namecrlt =('ID COHEN-COON')
        print(f'{Des}{namecrlt}')
        
if metodo == 5: #Asignacion polos
    if control == 1: 
        kp=K
        ti=0
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('ASIGNACION POLOS')
        print(f'{Des}{namecrlt}')
    else: #PID
        kp=0
        ti=0
        td=0
        incontrolabilidad = theta / tau  
        print(incontrolabilidad)
        namecrlt =('D ASIGNACION POLOS')
        print(f'{Des}{namecrlt}')

#Calculo del controle PID digital
q0=kp*(1+Ts/(2*ti)+td/Ts)
q1=-kp*(1-Ts/(2*ti)+(2*td)/Ts)
q2=(kp*td)/Ts

#Lazo Cerrado de Control
for k in range(nit):
    
    #=====================================================#
    #============    SIMULAR EL PROCESO REAL  ============#
    #=====================================================#
    
    #Con el Modelo NO LINEAL
    if k > 1:
        #lsim
        T1 = labtool.temperature_tclab(t[0:k+1], u[0:k+1] - q[0:k+1], Ta[0:k+1], Tinit)
        y[k] = T1[-1]
        
    #=====================================================#
    #============       CALCULAR EL ERROR     ============#
    #=====================================================#
    
    e[k] = r[k] - y[k]
    
    #=====================================================#
    #===========   CALCULAR LA LEY DE CONTROL  ===========#
    #=====================================================#
    #bias = (y[k] - y[0]) /kss
    u[k] = u[k-1] + q0*e[k] + q1*e[k-1] + q2*e[k-2]
    #u[k] = Kc*e[k] + bias
    #saturacion
    if u[k] > 100:
        u[k]  = 100
    elif u[k] < 0:
        u[k] = 0

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,r,'--k',t,y,'r-',linewidth=3)
plt.legend(['Setpoint', 'Output'])
plt.ylabel('Temperature (°C)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)
plt.title(_control + namecrlt,fontsize=22)

plt.subplot(2,1,2)
plt.step(t,u,'b-',linewidth=3)
plt.legend(['Heater'])
plt.ylabel('Power (%)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)
plt.show()
labtool.save_txt(t, u, y, r, Des + namecrlt)
plt.savefig(Des + namecrlt +".png")
plt.pause(1)