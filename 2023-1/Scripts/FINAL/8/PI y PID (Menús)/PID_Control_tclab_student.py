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
import tclab_cae.tclab_cae as tclab
import time
from datetime import date
#Importar funciones modelo NO Lineal
from Funtions import * 

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
Ts = 26                  #Periodo de Muestreo

#Función de transferencia continua
K = 1.84
theta1 = 35.2339
tau = 115.6
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
tsim = 999                #Tiempo de simulacion (segundos)
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

Des = f'Escogio el metodo P'
_control = f'Real P'

if metodo == 1: #Ziegler-Nichols
    if control == 1: # P
        kp = tau/(K*theta)
        ti = np.infty
        td = 0
        namecrlt = (' Control Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    elif control == 2: # PI
        kp = (0.9*tau)/(K*theta)
        ti = 3.3*theta
        td = 0
        namecrlt = ('I Control Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    else: # PID
        kp = ((1.2*tau)/(K*theta))
        ti = 2*theta
        td = 0.5*theta
        namecrlt =('ID Control Ziegler-Nichols')
        print(f'{Des}{namecrlt}')

elif metodo == 2: # IAE Roveri
    if control == 1: # P
        kp = 1
        ti = np.infty
        td = 0
        namecrlt =(' Control IAE Rovira')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        IAE_PI = (0.984, -0.986, 0.608, -0.707)
        kp = (1/K)*(IAE_PI[0]*(theta/tau)**IAE_PI[1])        
        ti = tau / (IAE_PI[2]*(theta/tau)**IAE_PI[3])
        td = 0
        namecrlt =('I Control IAE Rovira')
        print(f'{Des}{namecrlt}')
    else: # PID
        IAE_PID = (1.435, -0.921, 0.878, -0.749, 0.482, 1.137)
        kp = (1/K)*(IAE_PID[0]*(theta/tau)**IAE_PID[1])        
        ti = tau / (IAE_PID[2]*(theta/tau)**IAE_PID[3])
        td = tau * (IAE_PID[4]*(theta/tau)**IAE_PID[5])
        namecrlt =('ID Control IAE Rovira')
        print(f'{Des}{namecrlt}')

elif metodo == 3: # IAET Roveri
    if control == 1: # P
        kp = 1
        ti = np.infty
        td = 0
        namecrlt =(' Control ITAE Roveri')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        IAET_PI = (0.859, -0.977, 0.674, -0.68)
        kp = (1/K)*(IAET_PI[0]*(theta/tau)**IAET_PI[1])        
        ti = tau / (IAET_PI[2]*(theta/tau)**IAET_PI[3])
        td = 0
        namecrlt =('I Control ITAE Roveri')
        print(f'{Des}{namecrlt}')
    else: # PID
        IAET_PID = (1.357, -0.947, 0.842, -0.738, 0.381, 0.995)
        kp = (1/K)*(IAET_PID[0]*(theta/tau)**IAET_PID[1])        
        ti = tau / (IAET_PID[2]*(theta/tau)**IAET_PID[3])
        td = tau * (IAET_PID[4]*(theta/tau)**IAET_PID[5])
        namecrlt =('ID Control IAET Roveri')
        print(f'{Des}{namecrlt}')

elif metodo == 4: #COHEN-COON
    if control == 1: # P
        kp = (1.03+(0.35*(theta/tau)))*(tau/(K*theta))
        ti = np.infty
        td = 0
        namecrlt =(' Control Cohen-Coon')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        COHEN_COON_PI = (0.9, 0.083, 1.27, 0.6)
        kp = (COHEN_COON_PI[0]+(COHEN_COON_PI[1]*(theta/tau)))*(tau/(K*theta))
        ti = (theta*(COHEN_COON_PI[0]+(COHEN_COON_PI[1]*(theta/tau)))) / (COHEN_COON_PI[2]+(COHEN_COON_PI[3]*(theta/tau)))
        td = 0
        namecrlt =('I Control Cohen-Coon')
        print(f'{Des}{namecrlt}')
    else: # PID
        COHEN_COON_PID = (1.35, 0.25, 0.54, 0.33, 0.5)
        kp = (COHEN_COON_PID[0]+(COHEN_COON_PID[1]*(theta/tau)))*(tau/(K*theta))*0.5
        ti = (theta*(COHEN_COON_PID[0]+(COHEN_COON_PID[1]*(theta/tau)))) / (COHEN_COON_PID[2]+(COHEN_COON_PID[3]*(theta/tau)))
        td = (COHEN_COON_PID[4]*theta) / (COHEN_COON_PID[0]+(COHEN_COON_PID[1]*(theta/tau)))
        namecrlt =('ID Control Cohen-Coon')
        print(f'{Des}{namecrlt}')

else: # PA
    if control == 1: # P
        kp = 1
        ti = np.infty
        td = 0
        namecrlt = (' Control Pole Assignment')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        # Objetivo al 50%
        Tss = (4*tau)*0.3
        M_p = 1
        ep = np.sqrt((np.log(M_p/100)**2) / (np.pi**2 + np.log(M_p/100)**2))
        wn = 3 / (ep*Tss)
        kp = ((2*ep*wn*tau)-1) / K
        ti = (kp*K) / ((wn**2)*tau)
        td = 0
        namecrlt = ('I Control Pole Assignment')
        print(f'{Des}{namecrlt}') 
    else:        
        kp = 1
        ti = np.infty
        td = 0
        namecrlt = ('ID Control Pole Assignment')
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
        plt.grid()
        # plt.xlabel('Time(s)',fontsize=18)
        if control == 1:        
            plt.title('P Control',fontsize=24)
        elif control == 2:
            plt.title('PI Control',fontsize=24)
        elif control == 3:
            plt.title('PID Control',fontsize=24)
            
        plt.subplot(2,1,2)
        plt.step(t[0:k],u[0:k],'b-',linewidth=3)
        plt.legend(['Heater'])
        plt.ylabel('Power (%)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.grid()
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
    lab.Q2(0)
    print('Shutting down')
    lab.close()
    # plt.savefig('CP_response.png') 
    # np.save_txt(t[0:k], y[0:k], u[0:k])
    save_txt(t[0:k], u[0:k], y[0:k], r[0:k], _control + namecrlt)
    plt.savefig(_control + namecrlt +".png")
finally:
    # Disconnect from Arduino
    # lab.Q1(0)
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
    save_txt(t, u, y, r, _control+ namecrlt)
    plt.savefig(_control + namecrlt +".png")
    # save_txt(tc, uc, yc, rc, name+".txt")
    # plt.savefig(name+".png")
