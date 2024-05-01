# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:59:49 2024
Algunas ideas sobre los famosos procesos gaussianos
@author: abierto
"""

import numpy as np
import matplotlib.pyplot as pl
def exponential_cov(x, y, params):
    """ esto es un kernel exponencial
        Simplemente para probar el juego
    """
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

def conditional(x_new, x, y, params):
    """ permite obtener la media y la varianza de las medidas
    en los puntos x_new, si conocemos el valor y de la medida en los
    puntos x
    """
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    return(mu.squeeze(), sigma.squeeze())

def predict(x, data, kernel, params, sigma, t):
    """predecimos los valores y_pred en el puntos a partir de los
    valores t en los puntos data (o eso creo)
    """
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    return y_pred, sigma_new



#vamos con el ejemplo
theta = [1,10] #estos son los parametros del kernel gaussiano

#supongamos que partimos de una media 0
sigma0 = exponential_cov(0,0,theta)

#ahora vamos a obtener un valor a patir de una distribución gausiana
#de media cero y varianza 1
#y = [np.random.normal(scale=sigma0)]

#lo asociamos a un punto en el que hemos medido. x = 1.
x = [2.]
y = [np.sin(x[0])]
print(y)
sigma1 = exponential_cov(x, x, theta)

x_pred = np.linspace(-3,3,1000)

predictions = [predict(i, x, exponential_cov,theta, sigma1, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
pl.errorbar(x_pred,y_pred,yerr=sigmas,capsize=0)
pl.plot(x_pred,y_pred)

#calculamos para otro punto pero ahora, obtemos la varianza condi
#cionada a lo que ya hemos medido (el punto 1)
m,s = conditional([-0.7], x, y, theta)
y2 = np.sin(-0.7)

x.append(-0.7)
y.append(y2)

sigma2 = exponential_cov(x,x,theta)

predictions = [predict(i, x, exponential_cov,theta, sigma2, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
pl.errorbar(x_pred,y_pred,yerr=sigmas,capsize=0)
pl.plot(x_pred,y_pred)

x_more = np.arange(-2.5,0.05,2.5)
x += x_more.tolist()


