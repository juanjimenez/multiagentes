#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:57:30 2025
Esto son solo ejemplos de uso. Para calcular el regresor hay que usar las
funciones de kriging_exp.py
@author: juan
"""

from kriging_exp import *

#Esta funciónn simplemente crea funciones gausianas, que utilizo para modelar
#los posibles campos.
def gaussian(x,var,x0,h):
    '''
    Espera que se le pase una matriz de posiciones x, las posiciones deben
    ir en columna. calcula un valor de la gausiana en los puntos x suministrados
    var es una matriz que nos permite modelar la forma de la gausiana. Tiene
    que ser definida positiva.
    x0 marca el centro de la gaussiana
    h es el valor de pico de la gausiana
    '''
    ndat = x.shape[0]
    y =np.zeros(ndat)
    for i in range(ndat):
        y[i] = h*np.exp(-(x[i]-x0)@var@(x[i]-x0))/np.sqrt(lina.det(lina.inv(var)))/(2*np.pi)
    return y


#Generamos unos puntos de medida aleatorios en un cuadrado de 10X10
xm = 20*np.random.rand(120,2)-10

#definimos los parámetros de un par de gausianas para componer una función con
#un par de máximos,
#posicion de centros
centro = np.array([-6,5])
centro1 = np.array([6,-5])
#valores máximos
h = 30
h1 = 50
#matrix de la gausiana
sen = np.array([[0.2,0.1],[0.1,0.2]])
sen1 = np.array([[0.1,0.08],[0.08,0.5]])

#tomamos medidas de la función en los puntos seleccionados
ym = gaussian(xm,sen,centro,h)+ gaussian(xm,sen1,centro1,h1) 


#nos creamos un mallado para poder dibujar la función y los resultados del regresor

x0 = np.linspace(-10.,10.,41)
x1 = x0.copy()
xm0,xm1 = np.meshgrid(x0,x1)

#aquí estoy simplemente apilando en una tercer eje los datos de las dos matrices
#del mallado (en esta tercera 'dimensión' tengo los pares de puntos (x0,x1)) del
#malllado
xmc = np.stack((xm0,xm1),axis=2)

#creamos matrices para guardar los resultados de las funciones sobre el mallado
ymc = np.zeros(xm0.shape)
ymf = ymc.copy()
sigphi2 = np.zeros(xm0.shape)

#la funcion param esta definida y explicada en kriging_exp.py
param = parametros(xm,ym,semivar,fun = None)

for i in range(xmc.shape[0]):
    #calculamos los valores de la regresion en todos los puntos del mallado
    ymc[i], sigphi2[i] = regresor(xmc[i],xm,param[0],param[1],param[2],semivar)
    ymf[i] = gaussian(xmc[i],sen,centro,h) + gaussian(xmc[i],sen1,centro1,h1)


fig = pl.figure()
#pinto la función original ymf, los puntos empleados para calcular el krigin y 
#el resultado de la interpolación ymc en el mismo dibujo
ax = fig.add_subplot(projection='3d')
ax.scatter(xm[:,0],xm[:,1],ym,marker='o')
ax.plot_wireframe(xm0,xm1,ymc,color='r')
#ax.plot_wireframe(xm0,xm1,ymf)

#tambien pinto la sigma para que veas el valor que da en los puntos del mallado
fig = pl.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.plot_wireframe(xm0,xm1,sigphi2)

#mas dibujitos en curva de nivel que se ven mas claros
pl.figure()
pl.contour(xm0,xm1,ymf,10)
pl.contour(xm0,xm1,ymc,10)
pl.scatter(xm[:,0],xm[:,1],marker='o')

# Para pintar la varianza como errorbar pero se ve muy sucio. si se 
#quiere usar es mejor comentar el plot de la función original
#for i in range(xmc.shape[0]):
#    ax.errorbar(xmc[i,:,0],xmc[i,:,1],ymc[i],sigphi2[i])