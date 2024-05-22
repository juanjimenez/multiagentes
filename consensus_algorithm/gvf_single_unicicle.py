# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:48:21 2024
Ejmplillo de sistema multiagente que converge a una treayectoria.
La dinamica de los agentes corresponde a u uniciclo
@author: abierto
"""

import numpy as np
from scipy.integrate import solve_ivp as sl
import matplotlib.pyplot as plt


###################funciones para definir la trayectoria######################
def fun(w):
    '''
    Esta función debe definir una curva en 3D que seguirán los agentes
    debe estar definida en funcion de la coordenada w (augmented field)
    '''
    #f1 = 15*np.sin(2*w)
    #f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
    #f3 = 3+5*np.cos(2*w)
    f1 = 10*np.sin(w) + 20*np.sin(2*w)
    f2 = 10*np.cos(w)-20*np.cos(2*w)
    f3 = -20*np.sin(3*w)
    return np.array([[f1], [f2], [f3]])

def dfun(w):
    '''
    Derivada primera de la función anterior, La defino explícitamente
    '''
    df1 = 10*np.cos(w) + 40*np.cos(2*w)
    df2 = -10*np.sin(w) + 40*np.sin(2*w)
    df3 = -60*np.cos(3*w)
    return np.array([[df1], [df2], [df3]])
    
def ddfun(w):
    '''
    Derivada segunda de la función fun, La defino explícitamente
    '''
    ddf1 = -10*np.sin(w) - 80*np.sin(2*w)
    ddf2 = -10*np.cos(w) + 80*np.cos(2*w)
    ddf3 = 180*np.sin(3*w)
    return np.array([[ddf1], [ddf2], [ddf3]])

#matriz lalpaciana de un grafo conexion en daisy chain
def grafo_L(N):
    '''
    Cada agente solo se relaciona con el anterior y el siguiente, formando 
    una cadena cerrada

    '''
    L =2*np.eye(N)
    sd = -np.ones([N-1])
    L = L + np.diag(sd,1)+np.diag(sd,-1)
    
    # Llenar las esquinas con -1
    L[0, -1] = -1
    L[-1, 0] = -1
    return L


def din_agen(t,x,K,kth,v,E):
    ''' 
    Esta funcion define la dinamica de un agente
    incluye la guia del campo y el efecto de la  cordinación
    Asumo que la dimensión de movimiento en R^3, por tanto (-1)^n = (-1)^3 = -1
    En general, salvo que se diga lo contrario cada columna de valores 
    corresponde a un agente y cada fila a una variable (x,y,z,w),theta)
    '''
   
    #c = -L@(x[3,:] - wstr)
    dp = np.zeros((5,1)) #guardar las derivadas de los estados excepto theta
    #calculamos el valor del campo guía en la posición del agente
    xi= x.reshape(5,1)
    p = xi[0:3]
    w = xi[3,0]
    f = fun(w)
    df = dfun(w)
    ddf = ddfun(w)
    Kphi = K@(p-f)
    Kdf  = K@df
    Xi = np.vstack((-df-Kphi,-1+Kphi.T@df))
    nXi = np.linalg.norm(Xi)
    Xib = Xi/nXi
    nXip = np.linalg.norm(Xi[0:2])
    bXipn = xi[0:2]/nXip
    dp[0] = v*np.cos(xi[4,0]) #x
    dp[1] = v*np.sin(xi[4,0]) #y
    dp[2] = v*Xi[2,0]/nXip    #z
    dp[3]=  v*Xi[3,0]/nXip    #w (la coordenada virtual)
    
        
    Jpf =np.hstack( \
        (np.vstack((-K,Kdf.T)),\
         np.vstack((-ddf + K@df,Kphi.T@ddf-np.ones((1,3))@K@df**2))))
        
    Jxip= ((np.eye(4)-Xib@Xib.T)@Jpf/Xib)[0:2,:]
    dp[4] = -(bXipn.T@E@Jxip@dp[0:4]/np.linalg.norm(Xib[0:2])\
            -kth*dp[0:2].T/v@E@bXipn)[0]#dot theta       
    return dp.reshape(5)
    
###########################Integracion del modelo#############################


#Parámetro de simulación 
#dimensiones del espacio 3D
 
K =np.diag([0.002,0.002,0.002]) #ganancias

kth = 1

v = 10 #velocidad fija de los uniciclos
#coordenadas iniciales de los robots 
#pos = np.random.rand(N, n)*100 #filas: dimensiones
E = np.array([[0,-1],[1,0]])                          #columnas: nº de robots
p = np.random.randint(-50, 51, size=(3))
theta = 2*np.pi*np.random.random((1))

#añadimos a la matriz de posiciones la coordenada virtual w 
w = 1 #ejemplo: todos valen 1
x = np.hstack((p,w,theta))
t = (0,200)
sol = sl(din_agen,t,x,method='LSODA',\
         args=(K,kth,v,E),max_step=0.2)

#representación gráfica 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.plot(sol.y[,:], sol.y[i+1,:], sol.y[i+2,:])
#     ax.scatter(sol.y[i,0],sol.y[i+1,0], sol.y[i+2,0], marker='o')
ax.plot(sol.y[0,:],sol.y[1,:],sol.y[2,:])
ax.plot(sol.y[0,0],sol.y[1,0],sol.y[2,0],'o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Trayectoria de los agenetes')
plt.show()
