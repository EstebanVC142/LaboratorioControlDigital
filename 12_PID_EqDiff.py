"""
Control PID - TCLAB
Función de Transferencia Continua:
                2.34 exp(-8.76s)
    G(s) =  ----------------------
                148.44s + 1
                
Función de Transferencia Discreta:
                 0.1114 z + 0.01138
G(z) = z^(-2) * ---------------------   Ts = 8
                    z - 0.9475
    
@author: Sergio A. Castaño Giraldo
"""
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
#import sys

#Importar funciones modelo NO Lineal
sys.path.append('../functions') 
#import tclab_fun as fun  

from menus import *


plt.close()
Ts   =  26                  #Periodo de Muestreo

#Función de transferencia continua
K = 1.80909
theta1 = 11.8264
tau = 205.9033
#Corrección del retardo para control discreto
theta = theta1 + Ts/2

# Función de Transferencia Discreta

numz =  np.array([0.02628, 0.3552])   #Numerador
denz =  np.array([1, -0.7938])        #Denominador
d    =  1           #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
print(Gz)

# Parametros del Modelo No Lineal
Ta = 28
Tinit = 28

#Crea los vectores
tsim = 2000                 #Tiempo de simulacion (segundos)
nit = int(tsim/Ts)          #Numero de Iteraciones
t = np.arange(0, (nit)*Ts, Ts)  #Tiempo
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
y[:] = Tinit
e = np.zeros(nit)           #Vector de error
q = np.zeros(nit)           #Vector de disturbio
q[40:] = 2

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[5:] = 40

# Control Proporcional
if metodo == 1: #Ziegler-Nichols
    if control == 1: #PI
        kp=(0.9*tau)/(K*theta)*0.2
        ti=theta/0.3
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
    else: #PID
        kp=(1.2*tau)/(K*theta)*0.2
        ti=2*theta
        td=0.5*theta
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
if metodo == 2: #IAE Rovira
    if control == 2: #PI
        kp=(1/K)*((0.758)*((theta/tau)**-0.861))*0.2
        ti=tau/(1.02+((theta/tau)*-0.3232))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
    else: #PID Rovira
        kp=(1/K)*((1.086)*((theta/tau)**-0.869))
        ti=tau/(0.740+((theta/tau)*-0.130))
        td=tau*(0.348*((tau/theta)**0.914))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
if metodo == 3: #IAET Roveri
    if control == 3: #PI
        kp=(1/K)*((0.586)*((theta/tau)**-0.916))
        ti=(tau/(0.796+((theta/tau)*-0.147)))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
    else: #PID
        kp=(1/K)*((0.586)*((theta/tau)**-0.916))*0.2
        ti=(tau/(0.796+((theta/tau)*-0.147)))
        td=tau*(0.308*((tau/theta)**0.929))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)

if metodo == 4: #COHEN-COON
    if control == 4: 
        kp=((0.9+(0.083*(theta/tau)))*(tau/K*theta))
        ti=(theta*(0.9+(0.083*theta/tau)))/(1.27+(0.6*theta/tau))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
    else: #PID
        kp=(1.35+(0.25*theta/tau))*(tau/K*theta)
        ti=(theta*(1.35+(0.25*theta/tau)))/(0.54+(0.33*theta/tau))
        td=(0.5*theta/(1.35+(0.25*theta/tau)))
        incontrolabilidad = theta / tau        
        print(incontrolabilidad)
if metodo == 5: #Asignacion polos
    if control == 5: 
        kp=K
        ti=0
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
    else: #PID
        kp=0
        ti=0
        td=0
        incontrolabilidad = theta / tau  
        print(incontrolabilidad)


#Calculo do controle PID digital
q0=kp*(1+Ts/(2*ti)+td/Ts)
q1=-kp*(1-Ts/(2*ti)+(2*td)/Ts)
q2=(kp*td)/Ts

#Lazo Cerrado de Control
for k in range(nit-1):
    
    #=====================================================#
    #============    SIMULAR EL PROCESO REAL  ============#
    #=====================================================#
    
    #Con el Modelo NO LINEAL
# =============================================================================
#     if k > 1:
#         T1 = fun.temperature_tclab(t[0:k], u[0:k] - q[0:k], Ta, Tinit)
#         y[k] = T1[-1]
# =============================================================================
    
    #Con el Modelo Lineal
    if k > 1:
        Tlin, tlin, Xlin = lsim(Gz,u[0:k+1] - q[0:k+1], t[0:k+1])
        y[k] = Tlin[-1] + Tinit #Agrega la condicion inicial
    
    #=====================================================#
    #============       CALCULAR EL ERROR     ============#
    #=====================================================#
    e[k]= r[k] - y[k]
    
    #=====================================================#
    #===========   CALCULAR LA LEY DE CONTROL  ===========#
    #=====================================================#
    #bias = (y[k]-y[0])/kss
    u[k] = u[k-1] + q0*e[k] + q1*e[k-1] + q2*e[k-2]
    if u[k] > 100:
        u[k] = 100
    elif u[k] < 0:
        u[k] = 0



plt.figure()
plt.subplot(2,1,1)
plt.plot(t,r,'--k',t,y,'r-',linewidth=3)
plt.legend(['Setpoint', 'Output'])
plt.ylabel('Temperature (C)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)
plt.title('Proportional Control',fontsize=24)

plt.subplot(2,1,2)
plt.step(t,u,'b-',linewidth=3)
plt.legend(['Heater'])
plt.ylabel('Power (%)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)
plt.show()


