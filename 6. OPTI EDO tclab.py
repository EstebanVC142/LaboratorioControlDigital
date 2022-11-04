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
    Tamb = 26.9 + 273.15   #Kelvin
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
        y = odeint(EDO_TCLAB, y0, ts, args=(Q[i],p))
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
        j = j + ((((ym[i]) - 273.15) - xreal[i])/xreal[i])**2
    return j
    
    

#importar los datos
data = np.loadtxt('data_teoria_modelo.txt', delimiter=',',skiprows=1)
t1 = data[:,0].T #Tomamos todas filas de la primera columna y transponemos a vector filas
xreal = data[:,2].T  #Leemos los datos de la segunda columna.
Q = data[:,1].T

#Condición inicial
x0 = 300
n = 599 #Nimber of second time points
alpha = 0.014
U = 5
p0 = [U, alpha]

#Tiempo de integración
t = np.linspace(0, n-1,n) #Time vector


T1 = calc_post(t, p0, x0)

#costo inicial
print(f'initial SSE objetive: {objetive(p0)}')

#optimizacion
solution = minimize(objetive, p0, method='SLSQP')
p = solution.x
print(f'Final SSE objetive: {objetive(p)}')
print(solution.message)

T = calc_post(t, p, x0)

#Graficamos los resultados
plt.plot(t[0:n], T1[0:n]-273.15,'b:',label=r'$T$ Initial guess', \
         linewidth = 2)
plt.plot(t[0:n], T[0:n]-273.15,'r:',label=r'$T$ Final guess', \
         linewidth = 2)
plt.plot(t, xreal,'y-',label=r'$T$ from data', \
         linewidth = 2)
plt.legend(loc='best')
plt.ylabel('Temperature (°C)', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
plt.grid()