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
import tclab_cae.tclab_cae as tclab
import time
from datetime import date
from menus import *
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
Ts = 47                  #Periodo de Muestreo

#Función de transferencia continua
K = 1.80909
theta1 = 11.8264
tau = 205.9033
#Corrección del retardo para control discreto
theta = theta1 + Ts/2

# Función de Transferencia Discreta
numz =  np.array([0.02428, 0.3552])   #Numerador
denz =  np.array([1, -0.7938])        #Denominador
d    =  1                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
print(Gz)

#===========================================================#
# Connect to Arduino
lab = tclab.TCLab_CAE()
start_time = time.time()
prev_time = start_time
#sleep time
sleep_max = 1.0
#===========================================================#

# Parametros del Modelo No Lineal
Ta = lab.T2  #lab.T1
Tinit = lab.T3

#Crea los vectores
tsim = 1500                 #Tiempo de simulacion (segundos)
nit = int(tsim/sleep_max)          #Numero de Iteraciones
t = np.zeros(nit)          #Tiempo

#Vectores del proceso Real
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
e = np.zeros(nit)           #Vector de error

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[Ts*2:] = 40


Des = f'Escogio el metodo PI'

if metodo == 1: #Ziegler-Nichols
    if control == 1: #PI
        kp=(0.9*tau)/(K*theta)*0.2
        ti=theta/0.3
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt = ('Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    else: #PID
        kp=(1.2*tau)/(K*theta)*0.2
        ti=2*theta
        td=0.5*theta
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('D Ziegler-Nichols')
        print(f'{Des}{namecrlt}')
        
if metodo == 2: #IAE Rovira
    if control == 1: #PI
        kp=(1/K)*((0.758)*((theta/tau)**-0.861))*0.2
        ti=tau/(1.02+((theta/tau)*-0.3232))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('IAE Rovira')
        print(f'{Des}{namecrlt}')
    else: #PID Rovira
        kp=(1/K)*((1.086)*((theta/tau)**-0.869))
        ti=tau/(0.740+((theta/tau)*-0.130))
        td=tau*(0.348*((tau/theta)**0.914))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('D IAE Rovira')
        print(f'{Des}{namecrlt}')
        
if metodo == 3: #IAET Roveri
    if control == 1: #PI
        kp=(1/K)*((0.586)*((theta/tau)**-0.916))
        ti=(tau/(0.796+((theta/tau)*-0.147)))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('IAET Roveri')
        print(f'{Des}{namecrlt}')
    else: #PID
        kp=(1/K)*((0.586)*((theta/tau)**-0.916))*0.2
        ti=(tau/(0.796+((theta/tau)*-0.147)))
        td=tau*(0.308*((tau/theta)**0.929))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('D IAET Roveri')
        print(f'{Des}{namecrlt}')
        
if metodo == 4: #COHEN-COON
    if control == 1: 
        kp=((0.9+(0.083*(theta/tau)))*(tau/K*theta))
        ti=(theta*(0.9+(0.083*theta/tau)))/(1.27+(0.6*theta/tau))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
        namecrlt =('COHEN-COON')
        print(f'{Des}{namecrlt}')
    else: #PID
        kp=(1.35+(0.25*theta/tau))*(tau/K*theta)
        ti=(theta*(1.35+(0.25*theta/tau)))/(0.54+(0.33*theta/tau))
        td=(0.5*theta/(1.35+(0.25*theta/tau)))
        incontrolabilidad = theta / tau        
        print(incontrolabilidad)
        namecrlt =('D COHEN-COON')
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


#Calculo do controle PID digital
q0=kp*(1+Ts/(2*ti)+td/Ts)
q1=-kp*(1-Ts/(2*ti)+(2*td)/Ts)
q2=(kp*td)/Ts
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
        


        #Con el Modelo Lineal
        #if k > Ts:
        #    ts = np.arange(0,k+Ts, Ts)
        #    T1 = labtool.temperature_tclab(t[0:k+1], u[0:k+1] - q[0:k+1], Ta[0:k+1], Tinit)
        #    ys[k] = Tlin[-1] + Tinit #Agrega la condicion inicial
        

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
        if t[k]%Ts == 0:
            u[k] =  u[k-1*Ts] + q0*e[k] + q1*e[k-1*Ts] + q2*e[k-2*Ts]
        #    us[k] = us[k-1] + q0*es[k] + q1*es[k-1] + q2*es[k-2]
        else:
            u[k] = u[k-1]
        #    us[k] = us[k-1]
        #saturacion
        if u[k] > 100:
            u[k]  = 100
        elif u[k] < 0:
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
        if control == 1:        
            plt.title('PI Control',fontsize=24)
        elif control == 2:
            plt.title('PID Control',fontsize=24)
            
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
    # Disconnect from Arduino
    lab.Q1(0)
    print('Shutting down')
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
    name = namecrlt + formatted_date
    tc = t[:k].copy()
    uc = u[:k].copy()
    yc = y[:k].copy()
    rc = r[:k].copy()
    labtool.save_txt(tc, uc, yc, rc, name+".txt")
    plt.savefig(name+".png")
