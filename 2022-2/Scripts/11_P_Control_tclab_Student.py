"""
Control Proporcional - TCLAB
Función de Transferencia Continua:
                1.80909 exp(-11.8264s)
    G(s) =  ----------------------
                205.9033s + 1
                
Función de Transferencia Discreta:
                 0.02428 z + 0.3552
G(z) = z^(-1) * ---------------------   Ts = 26.70049
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
#sys.path.append('../functions') 
#import tclab_fun as fun  

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
# Función de Transferencia Discreta (Profesor)
Ts   =  47                  #Periodo de Muestreo
numz =  np.array([0.02428, 0.3552])   #Numerador
denz =  np.array([1, -0.7938])        #Denominador
d    =  1             #Retardo
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
Ta = lab.T1  #lab.T1
Tinit = lab.T1

#Crea los vectores
tsim = 600                 #Tiempo de simulacion (segundos)
nit = int(tsim/1)          #Numero de Iteraciones
t = np.zeros(nit)          #Tiempo


#Vectores del proceso Real
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
y[:] = Tinit
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
Kc = 2
bias = 6.4
kss = dcgain(Gz)  #Ganancia del sistema

#Crear plot
plt.figure(figsize=(10,7))
plt.ion() #Enable interactive mode
plt.show()

try:
    #Lazo Cerrado de Control
    for k in range(nit-1):
        
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
        
        #ys[k] = 0
        
        #Con el Modelo Lineal
        if k > Ts:
            ts = np.arange(0,k+Ts,Ts)
            Tlin, tlin, Xlin = lsim(Gz,us[0:k+Ts:Ts], ts)
            ys[k] = Tlin[-1] + Tinit # agregamos condicion inicial
        
        #=====================================================#
        #============       CALCULAR EL ERROR     ============#
        #=====================================================#
        e[k]= r[k] - y[k]
        es[k]= r[k] - ys[k]
        
        #=====================================================#
        #===========   CALCULAR LA LEY DE CONTROL  ===========#
        #=====================================================#
        #bias = (y[k] - y[0]) /kss
       


        if t[k]%Ts == 0:
            u[k] = Kc * e[k] + bias
            us[k] = Kc * es[k] + bias
        else:
            u[k] = u[k-1]
            us[k] = us[k-1]
        
            #saturamos la salida
        if u[k] >100:
            u[k] = 100
        elif u[k] <0:
            u[k] =0

            #saturamos la salida simulada
        if us[k] >100:
            us[k] = 100
        elif us[k] <0:
            us[k] =0
            
        #Mandar la Señal a la Planta
        lab.Q1(u[k])
        
        #Graficar
        plt.subplot(2,1,1)
        plt.plot(t[0:k],r[0:k],'--k',t[0:k],ys[0:k],'m--',\
                 t[0:k],y[0:k],'r-',linewidth=3)
        plt.legend(['Setpoint', 'Simulation', 'Output'])
        plt.ylabel('Temperature (C)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.title('Proportional Control',fontsize=24)

        plt.subplot(2,1,2)
        plt.step(t[0:k],us[0:k],'m--',\
                 t[0:k],u[0:k],'b-',linewidth=3)
        plt.legend(['Simulation', 'Heater'])
        plt.ylabel('Power (%)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.draw()
        plt.pause(0.05)
            
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

