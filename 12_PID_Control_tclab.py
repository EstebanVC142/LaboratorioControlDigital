# -*- coding: utf-8 -*-
"""
Control Proporcional - TCLAB
Función de Transferencia Continua:
                1.80909 exp(-11.8264s)
    G(s) =  ----------------------
                205.9033s + 1
                
Función de Transferencia Discreta:
                 0.02428 z + 0.3552
G(z) = z^(-1) * ---------------------   Ts = 47
                    z^2 - 0.7938 z
    
@authors: Juan David , Esteban, Andres
"""
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import sys
import tclab_cae.tclab_cae as tclab
import time

#Importar funciones modelo NO Lineal
sys.path.append('../functions') 
#import tclab_fun as fun  
from menus import *

def save_txt(t, u1, y1):
    data = np.vstack( (t, u1, y1) ) #Vertical stack
    data = data.T
    top = 'Time (sec),  Heater (%),  Temperature (C)'
    np.savetxt('data.txt', data, delimiter=',',header = top, comments='')

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
Ts   =  26              #Periodo de Muestreo
numz =  np.array([0.02428, 0.3552])   #Numerador
denz =  np.array([1, -0.7935])        #Denominador
d    =  1                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
print(Gz)

#Función de transferencia continua
K = 1.80909
theta1 = 11.8264
tau = 205.9033
#Corrección del retardo para control discreto
theta = theta1 + Ts/2


#===========================================================#
# Connect to Arduino
lab = tclab.TCLab_CAE()
start_time = time.time()
prev_time = start_time
#sleep time
sleep_max = 1.0
#===========================================================#

# Parametros del Modelo No Lineal
Ta = lab.T3  #lab.T1
Tinit = lab.T1

#Crea los vectores
tsim = 400              #Tiempo de simulacion (segundos)
nit = int(tsim/1)          #Numero de Iteraciones
t = np.zeros(nit)          #Tiempo

#Vectores del proceso Real
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
e = np.zeros(nit)           #Vector de error


#Vectores del proceso Simulado
us = np.zeros(nit)           #Vector de entrada (Heater)
ys = np.zeros(nit)           #Vector de salida  (Temperatura)
ys[:] = Tinit
es = np.zeros(nit)           #Vector de error

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[Ts:] = 40


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
        ti=tau/(1.02+((theta/tau)*-0.323))
        td=0
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
    else: #PID Rovira
        kp=(1/K)*((1.086)*((theta/tau)**-0.869))*0.2
        ti=tau/(0.740+((theta/tau)*-0.130))
        td=tau*(0.348*((tau/theta)**0.914))
        incontrolabilidad = theta / tau
        print(incontrolabilidad)
if metodo == 3: #IAET Roveri
    if control == 3: #PI
        kp=(1/K)*((0.586)*((theta/tau)**-0.916))*0.2
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
        kp=0
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
#Calculondo controle PID digital
q0=kp*(1+Ts/(2*ti)+td/Ts)
q1=-kp*(1-Ts/(2*ti)+(2*td)/Ts)
q2=(kp*td)/Ts

#Crear plot
plt.figure(figsize=(10,7))
plt.ion() #Enable interactive mode
plt.show()

try:
    #Lazo Cerrado de Control
    for k in range(nit- 1):
        
        tm = delay_time(sleep_max, prev_time)
        prev_time = tm
        t[k] = np.round(tm - start_time)
        
        #=====================================================#
        #============         PROCESO REAL        ============#
        #=====================================================#
        y[k] = lab.T1
        
        #=====================================================#
        #============    SIMULAR EL PROCESO REAL  ============#
        #=====================================================#
        
        #Con el Modelo NO LINEAL
# =============================================================================
#         if k > 1:
#             T1 = fun.temperature_tclab(t[0:k], u[0:k] - q[0:k], Ta, Tinit)
#             y[k] = T1[-1]
# =============================================================================
        
        #Con el Modelo Lineal
        if k > Ts:
            ts = np.arange(0,k+Ts, Ts)
            Tlin, tlin, Xlin = lsim(Gz, us[0:k+Ts:Ts], ts)
            ys[k] = Tlin[-1] + Tinit #Agrega la condicion inicial
        
        #=====================================================#
        #============       CALCULAR EL ERROR     ============#
        #=====================================================#
        e[k]= r[k] - y[k]
        es[k]= r[k] - ys[k]
        
        #=====================================================#
        #===========   CALCULAR LA LEY DE CONTROL  ===========#
        #=====================================================#
        #bias = (y[k]-y[0])/kss
        if t[k]%Ts == 0:
            u[k] =  u[k-1] + q0*e[k] + q1*e[k-1] + q2*e[k-2]
            us[k] = us[k-1] + q0*es[k] + q1*es[k-1] + q2*es[k-2]
        else:
            u[k] = u[k-1]
            us[k] = us[k-1]
            
        if u[k] > 100:
            u[k] = 100
        elif u[k] < 0:
            u[k] = 0
        
        
        if us[k] > 100:
            us[k] = 100
        elif us[k] < 0:
            us[k] = 0
        
        # write Heater (0 -100)
        lab.Q1(u[k])
        
        #Graficar
        plt.subplot(2,1,1)
        plt.plot(t[0:k],r[0:k],'--k',t[0:k],ys[0:k],'m--',\
                 t[0:k],y[0:k],'r-',linewidth=3)
        plt.legend(['Setpoint', 'Simulation', 'Output'])
        plt.ylabel('Temperature (C)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.title('PID Control',fontsize=24)

        plt.subplot(2,1,2)
        plt.step(t[0:k],us[0:k],'m--',\
                 t[0:k],u[0:k],'b-',linewidth=3)
        plt.legend(['Simulation', 'Heater'])
        plt.ylabel('Power (%)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.draw()
        plt.pause(0.05)
        plt.savefig('PRUEBznpid-PID.png')
        plt.show()
            
    # Turn off heaters
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    lab.close()
    plt.savefig('CP_response.png') 
    save_txt(tm[0:k], y[0:k], u[0:k])

# Allow user to end loop with Ctrl-C          
except KeyboardInterrupt:
    # Disconnect from Arduino
    lab.Q1(0)
    print('Shutting down')
    lab.close()
    plt.savefig('CP_response.png') 
    save_txt(tm[0:k], y[0:k], u[0:k])
       
except:
    # Disconnect from Arduino
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    lab.close()
    print('Shutting down')
    plt.savefig('CP_response.png') 
    save_txt(tm[0:k], y[0:k], u[0:k])
    raise

