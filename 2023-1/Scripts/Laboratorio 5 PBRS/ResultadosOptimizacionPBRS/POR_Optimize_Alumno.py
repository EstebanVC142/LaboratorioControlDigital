import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# Ecuación Diferencial de Primer Orden con Retardo    
def Fopdt(t,y,uf,Km,taum,thetam):
    # argumentos
    #  y      = Salida
    #  t      = Tiempo
    #  uf     = Función de entrada lineal (Para hacer desplazamiento temporal)
    #  Km     = Ganancia
    #  taum   = Constante de Tiempo
    #  thetam = Retardo
    
    try:
        if (t-thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t-thetam)
    except:
        um = u0
    # Calcula la Derivada
    dydt = (-(y-yp0) + Km * (um-u0))/taum
    return dydt

# Simulación del sistema de primer orden con x=[Km,taum,thetam]
def SimulacionModelo(x):
    # Argumentos de entrada
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    # Vector de Salida
    ym = np.ones(ns) * yp0  # model
    # Condición Inicial
    y0 = [yp0]
    # Simulación del Modelo (Integración)   
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
        sol = solve_ivp(Fopdt,ts, y0,args=(uf,Km,taum,thetam))
        y0 = sol.y[:, -1]
        ym[i+1] = y0[0]
    return ym

# Función Objetivo
def objective(x):
    '''Recibe los  y_reales y los y_simulados
    Busca estimar desde un punto inicial los parametros
    K, tau, theta. se cambian dichos valores para que en 
    en algun punto el comportamiento dinamico de la FT
    tenga una funcion de costo baja comparada con los datos
    reales.'''
    # Calcula el costo
    obj = 0.0
    #Simular el Modelo
    ym = SimulacionModelo(x)
    for i in range(len(ym)):
        obj = obj + ((yp[i] - ym[i]) / yp[i]) ** 2

    return obj

# Columna 0 = tiempo (t)
# Columna 1 = Entrada (u - Heater)
# Columna 2 = Salida (yp - Temperatura)

data = np.loadtxt('D:/Esteban VC/Poli JIC/Semestres/2023-1/Control Analogo y Digital/LaboratorioControlDigital/2023-1/Scripts/Laboratorio 5 PBRS/Lab 7 - PRBS - Discretizacion/prbsResponse_25.txt',delimiter=',',skiprows=1)

# Entrada y Salida Inicial
u0 = data[0,1]
yp0 = data[0,2]

#Datos del experimeto por PRBS
t = data[:,0].T - data[0,0]
u = data[:,1].T
yp = data[:,2].T

# Numero de Iteraciones
ns = len(t)
# Interpolación lineal del tiempo con u (util para involucrar el retardo)
uf = interp1d(t,u)

# Estimativa Inicial
K = 1 #Aproximacion de los datos para que no se demore tanto jaja, los dejé en 0 y llevaba mas de una hora sin acabar
tau = 100
theta = 10

''' Parametros finales: 
                        Costo inicial: 453.0715662520367
                        Costo final: 0.07345667273220106
                        kp: 3.3367868778032426
                        tau: 239.62203109492359
                        theta: 9.961170486841828
'''

x0 = np.zeros(3)
x0[0] = K # Km
x0[1] = tau # taum
x0[2] = theta # thetam

# OPTIMIZACION
print(f'Costo inicial: {objective(x0)}')

# Metodo de optimización por minimos cuadrados
# Varia los 3 parametros para que acerce al comportamiento real.
solucion = minimize(objective, x0)
x = solucion.x #Guardo mis 3 parametros ya optimizados

print(f'Costo final: {objective(x)}')
print('kp: ' + str(x[0]))
print('tau: ' + str(x[1]))
print('theta: ' + str(x[2]))

#Simulación del modelo con los 3 parametros iniciales
ym1 = SimulacionModelo(x0) 
#Simulación del modelo con los 3 parametros Finales
ym2 = SimulacionModelo(x)

plt.figure(figsize=(10,7))
plt.title('PBRS Optimizado', fontsize=16)
plt.plot(t, yp, 'k', linewidth=2, label='Datos Proceso')
plt.plot(t, ym1, 'b-', linewidth=2, label='Estimativa Inicial')
plt.plot(t, ym2, 'r--', linewidth=2, label='FT POR Optimizada')
plt.ylabel('Temperatura [°C]', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=14)
plt.legend(loc= 'best')
plt.grid()
plt.show()
plt.savefig('prbs_Optimized.png')
