# -*- coding: utf-8 -*-
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

#Importar funciones modelo NO Lineal
#sys.path.append('../functions') 
#import tclab_fun as fun  

##.02428
plt.close()
# Función de Transferencia Discreta (Profesor)
Ts   =  47                  #Periodo de Muestreo
#umz =  np.array([0.01332, 0.3573])   #Numerador
#denz =  np.array([1, -0.7986,0])        #Denominador
numz =  np.array([0.02428, 0.3552])   #Numerador
denz =  np.array([1, -0.7938])        #Denominador
d    =  1           #Retardo
denzd = np.hstack((denz, np.zeros(d)))
print(denzd)
Gz   =  tf(numz, denzd, Ts)
print(Gz)

# Parametros del Modelo No Lineal
Ta = 23
Tinit = 23

#Crea los vectores
tsim = 2000                #Tiempo de simulacion (segundos)
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
Kc = 2
bias = 6.4
kss = dcgain(Gz)  #Ganancia del sistema

print(kss)

#Lazo Cerrado de Control
for k in range(nit):
    
    #=====================================================#
    #============    SIMULAR EL PROCESO REAL  ============#
    #=====================================================#
    
    #Con el Modelo NO LINEAL
    #y[k] = 0
    
    #Con el Modelo Lineal
    if k > 1:
        Tlin, tlin, Xlin = lsim(Gz,u[0:k+1] - q[0:k+1], t[0:k+1])
        y[k] = Tlin[-1] + Tinit # agregamos condicion inicial

    #=====================================================#
    #============       CALCULAR EL ERROR     ============#
    #=====================================================#
    e[k]= r[k] - y[k]
    
    #=====================================================#
    #===========   CALCULAR LA LEY DE CONTROL  ===========#
    #=====================================================#
    #bias = (y[k] - y[0]) /kss
    u[k] = Kc * e[k] + bias
    
    #saturamos la salida
    if u[k] > 100:
        u[k] = 100
    elif u[k] < 0:
        u[k] = 0
    #Agrega el disturbioo de entrada
    u[k] = u[k] - q [k]
    


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


