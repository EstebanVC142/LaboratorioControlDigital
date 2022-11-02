from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Programamos la EDO
# x: entrada de ecuacion diferencial
# t: Tiempo de integracion
# u: Parametro de entrada (Q de nuestro transistor)
def EDO_TCLAB(x, t, Q, p):
    
    U = p[0] # starting at 10.0 W/m^2-K
    alpha = p[1] # starting as 0.01 W / % heater
             
    Tamb = 296.15   #Kelvin
    A = 0.0012      #Transistor area
    m = 0.004       #Transistor mass
    Cp = 500        #heat capacity
    ε = 0.9         #Emissivity
    sigma = 5.67e-8        #Stefan-boltzman constant     
    
    #Renombrando los estados de EDO
    T = x[0]
        
    #Nonlinear Energy Balance
    dTdt = ((U*A*(Tamb-T)/(m*Cp)) \
            + ε * sigma * A * (Tamb**4 - T**4) \
            + alpha*Q)

    return dTdt

#calculo del modelo
def calc_post(t, p, xo):
    y0 = x0
    # Inicializamos todo el vector con uno
    # que tiene la misma longuitud del tiempo 
    # multiplicado por la condición inical.

    ym = np.ones(len(t1)) * y0
    for i in range(len(t1) - 1):
        ts = [t[i],t[i+1]] # definimos la integración en pequeños instantes
        y = odeint(EDO_TCLAB, y0, ts, args=(Q,p0))
        y0 = y[-1]
        ym[i+1] = y0[0]
    return ym
        
        
#Define objetive
#llamamos el integrador para que la función resuelva conforme a los parametros ingresados
def objetive(p):
    #simulate model
    ym = calc_post(t, p, x0)
    # calculate objetive
    j=0.0
    for i in range(len(t1)-1):
        j = j + ((ym[i] - xreal[i])/xreal[i])**2
    return j
    
    

#importar los datos
data = np.loadtxt('data.txt', delimiter=',',skiprows=1)
t1 = data[:,0].T #Tomamos todas filas de la primera columna y transponemos a vector filas
xreal = data[:,2].T  #Leemos los datos de la segunda columna.


#Condición inicial
x0 = 300.0
Q = 40.0 # percent heater (0 - 100%)
n = 600 + 1 #Nimber of second time points
alpha = 0.014
U = 5
p0 = [U, alpha]

# Percent Heater (0-100%)
Q1 = np.zeros(n)

#Tiempo de integración
t = np.linspace(0, n-1,n) #Time vector

#Solución de ecuación diferencial
T1 = odeint(EDO_TCLAB, x0, t, args=(Q,p0)) #Integrate ODE

#costo inicial
print(f'initial SSE objetive: {objetive(p0)}')

#Graficamos los resultados




#optimizacion
bnds = ((2.0,20.0),(0.005,0.05))
solution = minimize(objetive, p0, method='SLSQP', bounds=bnds)
p = solution.x
print(f'Final SSE objetive: {objetive(p)}')
print(solution.message)


T = odeint(EDO_TCLAB, x0, t, args=(Q,p)) #Integrate ODE

#Graficamos los resultados

plt.plot(t[0:n], T1[0:n]-273.15,'b:',label=r'$T$ Initial guess', \
         linewidth = 2)
plt.plot(t[0:n], T[0:n]-273.15,'g:',label=r'$T$ Final guess', \
         linewidth = 2)
plt.plot(t1, xreal,'y-',label=r'$T$ from data', \
         linewidth = 2)
plt.legend(loc='best')
plt.ylabel('Temperature (°C)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
plt.grid()

'''

#Graficamos los resultados
ax = plt.subplot(211)
ax.grid()
plt.plot(t[0:n], T[0:n]-273.15,'b:',label=r'$T$ Initial guess', \
         linewidth = 2)
plt.plot(t1, xreal,'r-',label=r'$T$ from data', \
         linewidth = 2)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
'''


'''
ax = plt.subplot(211)
ax.grid()
plt.plot(t[0:n], T[0:n]-273.15,'-k',label=r'$T$ energy balance', \
         linewidth = 2)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
plt.legend(loc='best')
plt.draw()
plt.pause(0.05)


plt.clf() #Clear current figure
ax = plt.subplot(212)
ax.grid()
plt.plot(t[0:n], T[0:n],'-r',label=r'$T$ energy balance', \
         linewidth = 2)
plt.ylabel('Temperature (K)', fontsize=14)
plt.legend(loc='best')

ax = plt.subplot(213)
ax.grid()
plt.plot(t[0:n], Q1[0:n],'-k',label=r'$Q$ ', \
         linewidth = 2)
plt.ylabel('Heater (%)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
plt.legend(loc='best')
plt.draw()
plt.pause(0.05)


'''