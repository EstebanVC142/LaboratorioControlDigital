    
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
#Importar funciones modelo NO Lineal (Elaborados por el grupo)
import Funtions as labtool  

plt.close()
# Función de Transferencia Discreta
Ts   =  47                  #Periodo de Muestreo
numz =  np.array([0.2895, 0.08482])   #Numerador
denz =  np.array([1, -0.7931])        #Denominador
d    =  1                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
print(Gz)

# Parametros del Modelo No Lineal
Ta = 27
Tinit = 27

#Crea los vectores
tsim = 1100               #Tiempo de simulacion (segundos)
nit = int(tsim/Ts)          #Numero de Iteraciones
t = np.arange(0, (nit)*Ts, Ts)  #Tiempo
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
y[:] = Tinit                #Inicializa el vector con Tinit
e = np.zeros(nit)           #Vector de error
q = np.zeros(nit)           #Vector de disturbio
q[40:] = 2

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[4:] = 40

# Control Proporcional
Kc = 1.8
bias = 0
kss = dcgain(Gz)  #Ganancia del sistema
print(f'kss={kss}')

#Lazo Cerrado de Control
for k in range(nit):
    #=====================================================#
    #============    SIMULAR EL PROCESO REAL  ============#
    #=====================================================#
    
    #Con el Modelo NO LINEAL
    if k > 1:
        
        T1 = labtool.temperature_tclab(t[0:k+1], u[0:k+1] - q[0:k+1], [Ta], Tinit)
        y[k] = T1[-1]       
     
    #=====================================================#
    #============       CALCULAR EL ERROR     ============#
    #=====================================================#
    
    e[k] = r[k] - y[k]
    
    #=====================================================#
    #===========   CALCULAR LA LEY DE CONTROL  ===========#
    #=====================================================#
    bias = (y[k] - y[0])/kss

    u[k] = Kc*e[k] + bias

    #satiracion
    if u[k] > 100:
        u[k]  = 100
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
labtool.save_txt(t, u, y, r,"SimulacionProporcional")
plt.savefig("Proportional Control.png")