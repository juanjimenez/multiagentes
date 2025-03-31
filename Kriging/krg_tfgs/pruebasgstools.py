#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:45:59 2025

@author: juan
"""

import numpy as np
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
import gstools as gs

s=9

#Campo aleatorio en 2D
np.random.seed(s)
L = 10
grid_size = 50
X = np.linspace(0,L,grid_size)
Y = np.linspace(0,L,grid_size)
XX, YY = np.meshgrid(X,Y)
lim = [0,L,0,L]
campo_real = 5*np.sin(0.5*XX)+2*np.cos(0.5*YY)+np.random.normal(0,0.05,XX.shape)

#Campo aletorio con gstools
model = gs.Gaussian(dim=2,var=10,len_scale=5)
srf = gs.SRF(model,seed=s)
campo_real = srf((X,Y),mesh_type='structured')

#Puntos iniciales
n_puntos = 10
puntos_x = np.random.rand(n_puntos)*L
puntos_y = np.random.rand(n_puntos)*L
puntos_z = [srf([puntos_x[i],puntos_y[i]],seed=s) for i in range(n_puntos)]
#puntos_z = np.array(puntos_z) ?

def variogram(param,h):
    var = param[0] 
    q = param[1]
    theta = param[2]
    cor = np.exp(-(abs(h)**q)/theta)
    return (var**2)*(1-cor)
#No sé como meter anisotropía en la función, ya que UniversalKriging toma el valor de h
#Hay que definir el variograma, no la covarianza
#No satura la varianza a (var**2)?

theta = 50
q = 2
sigma = 0.5
uk = UniversalKriging(
    puntos_x,puntos_y,puntos_z,
    variogram_model="custom",variogram_parameters=[theta,q,sigma],variogram_function=variogram,
    drift_terms=["regional_linear"]
)
campo_estimado,varianza_estimada = uk.execute("grid",X,Y)


#Graficar los resultados
fig,axes = plt.subplots(1,3,figsize=(15,5))

#Al ponerlo en los ejes
#Campo Real
ax = axes[0]
im = ax.imshow(np.transpose(campo_real),extent=lim,origin="lower")
ax.scatter(puntos_x,puntos_y,color="white",marker="o",label="Puntos de medición")
ax.set_title("Campo Real")
plt.colorbar(im)
ax.legend()

#Campo estimado
ax = axes[1]
im = ax.imshow(campo_estimado,extent=lim,origin="lower")
ax.scatter(puntos_x,puntos_y,color="white",marker="o",label="Puntos de medición")
ax.set_title("Campo Estimado")
plt.colorbar(im)
ax.legend()

#Varianza estimada
ax = axes[2]
im = ax.imshow(varianza_estimada,extent=lim,origin="lower")
ax.scatter(puntos_x,puntos_y,color="white",marker="o",label="Puntos de medición")
ax.set_title("Varianza Estimada")
plt.colorbar(im)
ax.legend()

plt.tight_layout()
plt.show()

#Comprobación de puntos
var_puntos = []
campo_puntos = []
for i in range(n_puntos):
    var_punto = uk.execute("points",[puntos_x[i]],[puntos_y[i]])[1][0]
    var_puntos.append(var_punto)
    campo_punto = uk.execute("points",[puntos_x[i]],[puntos_y[i]])[0][0]
    campo_puntos.append(campo_punto)

print('Campo real')
[print('El valor del campo en el punto ',[puntos_x[i],puntos_y[i]],' es ',puntos_z[i]) for i in range(n_puntos)]
print()

print('Campo estimado')
[print('El valor del campo en el punto ',[puntos_x[i],puntos_y[i]],' es ',campo_puntos[i]) for i in range(n_puntos)]
print(f"Campo mínimo: {np.min(campo_estimado)}, máximo: {np.max(campo_estimado)}")
print()

print('Varianza estimada')
[print('El valor del campo en el punto ',[puntos_x[i],puntos_y[i]],' es ',var_puntos[i]) for i in range(n_puntos)]
print(f"Varianza mínima: {np.min(varianza_estimada)}, máxima: {np.max(varianza_estimada)}")