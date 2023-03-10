import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# Columna 0 = tiempo (t)
# Columna 1 = Entrada (u - Heater)
# Columna 2 = Salida (yp - Temperatura)
data = np.loadtxt('data_PRBS_30.txt',delimiter=',',skiprows=1)

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

# Ecuación Diferencial de Primer Orden con Retardo    
def fopdt(y,t,uf,Km,taum,thetam):
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
    # y para nuestro caso es la temperatura
    # yp0 condiciones iniciales de los datos
    return dydt

# Simulación del sistema de promer orden con x=[Km,taum,thetam]
def sim_model(x):
    # Argumentos de entrada
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    # Vector de Salida
    ym = np.zeros(ns)  # model
    # Condición Inicial
    ym[0] = yp0
    # Simulación del Modelo (Integración)   
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
        ym[i+1] = y1[-1]
    return ym

# Función Objetivo
def objective(x):
    #Recibe los  y_reales y los y_simulados
    #Busca estimar desde un punto inicial los parametros
    #K, tau, theta. se cambian dichos valores para que en 
    #en algun punto el comportamiento dinamico de la FT
    #tenga una funcion de costo baja comparada con los datos
    #reales.
    # Calcula el costo
    obj = 0.0
    #Simular el modelo
    ym = sim_model(x)
    #ym = modelo
    #yp = planta
    for i in range(len(ym)): #Hasta el tamaño de ym
        obj = obj + (ym[i] - yp[i])**2
        
    return obj

# Estimativa Inicial
K = 2.031
tau = 195.4
theta = 22.6334

x0 = np.zeros(3)
x0[0] = K # Km
x0[1] = tau # taum
x0[2] = theta # thetam

print(f'costo inicial {objective(x0)}')

# Metodo de optimización por minimos cuadrados
# Varia los 3 parametros para que acerce al comportamiento
# real.
solucion = minimize(objective, x0)
x = solucion.x #Guardo mis 3 parametros ya optimizados



print(f'costo final {objective(x)}')
print('kp:' + str(x[0]))
print('tau:' + str(x[1]))
print('theta:' + str(x[2]))

#Simulación del modelo con los 3 parametros iniciales
ym1 = sim_model(x0) 
#Simulación del modelo con los 3 parametros Finales
ym2 = sim_model(x)

plt.figure()
plt.plot(t, yp, 'b', linewidth=2, label=' Datos proceso')
plt.plot(t, ym2, 'y--', linewidth=2, label=' Estimativa Final')
plt.plot(t, ym1, 'r--', linewidth=2, label=' Estimativa Inicial')
plt.ylabel('Temperatura')
plt.xlabel('Tiempo')
plt.legend(loc = 'best')
plt.grid()
plt.show()