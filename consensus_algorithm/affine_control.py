#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:23:01 2021

@author: juanjimenez
playing with consesus algorithm
"""
import numpy as np
import graph_utils
from matplotlib import pyplot as pl
from scipy import linalg as lng

N = 6 #number of agents

#graph building 
V = range(N) #vertices
#Undirected generically universally rigid graph (hopefully) 
E = [(0,1),(0,3),(0,5),(1,2),(1,4),(2,3),(4,5)]

w = 1. #weight for the edges (equal for all edges)

B = graph_utils.incidence(V,E)
L = graph_utils.Laplacian(w,B)

print('incidence\n',B,'\n','laplacian\n',L)


#a configuration of interest
pcm = np.array([[0.],[0.]]) #centroid
uno = np.ones([N,1]) #the staked '1'
# Desires Position of agents, (axis located at centroid)
pstar0 = np.array([[0.],[1.]])
pstar1 = np.array([[-1.],[0.]])
pstar2 = np.array([[-1.],[-1.]])
pstar3 = np.array([[0.],[-1.]])
pstar4 = np.array([[1.],[0.]])
pstar5 = np.array([[1.],[1.]])
pstar =np.kron(uno,pcm)+\
np.concatenate((pstar0,pstar1,pstar2,pstar3,pstar4,pstar5),axis=0)
for i in range(0,2*N,2):
    pl.plot(pstar[i],pstar[i+1],'x')

#Stacked Incidence matrix
B_bar = np.kron(B,np.eye(2))
#Stacked Laplacian
L_bar = np.kron(L,np.eye(2))


#initial random positions
p = np.random.rand(12,1)*3
for i in range(0,2*N,2):
    pl.plot(p[i],p[i+1],'.')

#Consensus algorithm    
step =0.01
t = 0

while t < 10:
    p = p -np.dot(L_bar,(p-pstar))*step
    for i in range(0,2*N,2):
        pl.plot(p[i],p[i+1],'.',markersize=1)
    t = t + step
    

#for i in range(0,2*N,2):
#    pl.plot(p[i],p[i+1],'.')   

pl.plot(p[0::2],p[1::2])

# zstar
#zstar = np.dot(B_bar.T,pstar)

#translation is trivial
# vstar = np.array([[1],[2]])
# vstar_stk = np.kron(uno,vstar)
# t = 0
# while t < 10:
#     p = p + (-np.dot(L_bar,(p-pstar))+vstar_stk)*step
#     for i in range(0,2*N,2):
#         pl.plot(p[i],p[i+1],'.',markersize=1)
#     t = t + step
# pl.plot(p[0::2],p[1::2])

# Basis for affine collective motions in 2D
A1 = np.array([[1,0],[0,0]])
A2 = np.array([[0,1],[0,0]])
A3 = np.array([[0,0],[1,0]])
A4 = np.array([[0,0],[0,1]])
# Now we can build whatever movement we want...

#rotation
W = -A2+A3

step =0.01
t = 0
W_stk = np.kron(np.eye(6),W)
while t < 50:
    p = p + (-np.dot(L_bar,(p-pstar))+np.dot(W_stk,p-pstar))*step
    for i in range(0,2*N,2):
        pl.plot(p[i],p[i+1],'.',markersize=1)
    t = t + step

pl.plot(p[0::2],p[1::2])
#vstart
#
#kappa = np.linalg.norm(vstar)
#vstar_n = vstar/kappa

#translation is trivial ¿?

#lng.solve()
#