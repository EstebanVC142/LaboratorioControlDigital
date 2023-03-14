from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

#Programamos la EDO
# x: entrada de ecuacion diferencial
# t: Tiempo de integracion
# u: Parametro de entrada (Q de nuestro transistor)
def EDO_TCLAB(x, t, Q):
    
    #Estamos abriendo el puerto serial
    U = 5           #heat transfer coefficient
    Tamb = 26.9 + 273.15   #Kelvin
    A = 0.0012      #Transistor area
    m = 0.004       #Transistor mass
    Cp = 500        #heat capacity
    ε = 0.9         #Emissivity
    sigma = 5.67e-8        #Stefan-boltzman constant
    alpha = 0.014       
    
    #Renombrando los estados de EDO
    T = x[0]
        
    #Nonlinear Energy Balance
    dTdt = (1.0/(m*Cp))*(U*A*(Tamb-T) \
            + ε * sigma * A * (Tamb**4 - T**4) \
            + alpha*Q)

    return dTdt

#Condición inicial
x0 = 300.0
Q = 40.0    # percent heater (0 - 100%)
n = 599  #Nimber of second time points

#Tiempo de integración
t = np.linspace(0, n-1,n) #Time vector

#Solución de ecuación diferencial
T = odeint(EDO_TCLAB, x0, t, args=(Q,)) #Integrate ODE EN KELVIN

# Lectura de datos
data = np.loadtxt('data_teoria.txt', delimiter=',',skiprows=1)
Q = data[:,1].T


#Graficamos los resultados
ax = plt.subplot(211)
ax.grid()

plt.clf() #Clear current figure
ax = plt.subplot(211)
ax.grid()
plt.plot(t[0:n], T[0:n]- 273.15,'-r',label=r'$T$ energy balance', \
         linewidth = 2)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend(loc='best')


ax = plt.subplot(212)
ax.grid()
plt.plot(t[0:n], Q[0:n],'-k',label=r'$Q$ ', \
         linewidth = 2)
plt.ylabel('Heater (%)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
plt.legend(loc='best')
plt.draw()
plt.pause(0.05)