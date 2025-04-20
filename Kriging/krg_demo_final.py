# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:44:04 2025
Playing with kriging and other fancy stuff
Calculado con los parametros del artículo para ver qué sale
@author: abierto
"""
from kriging_util import *
#nos vamos a generar un campo y sus resultados en una malla formada
#por x e y
x = y = np.arange(-25.,25.2,0.2)
xm,ym = np.meshgrid(x,y)
V = np.zeros(xm.shape)
mogo = np.stack((xm,ym),axis=2)
theta = np.pi/6
sigma = np.array([[20.,0.],[0,80.]])
sigmar,norm =rotaynorm(theta,sigma)
mu = np.array([10.,10.])
 
sigma1 = np.array([[100,0],[0,100]])
theta1 = np.pi/2
sigmar1,norm1 =rotaynorm(theta1,sigma1)
mu1 = np.array([0,-18])

sigma2 = np.array([[20,0],[0,20]])
theta2 = 0
sigmar2,norm2 =rotaynorm(theta2,sigma2)
mu2 = np.array([-10,18])

for i in range(mogo.shape[0]):
    for j in range(mogo.shape[1]):
        V[i,j] = 1000*gausianilla(mogo[i,j,:],sigmar,mu,norm) + \
        2000*gausianilla(mogo[i,j,:],sigmar1,mu1,norm1) +\
        100*gausianilla(mogo[i,j,:],sigmar2,mu2,norm2)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_wireframe(xm,ym,V)
plt.xlabel('x')


#ahora viene lo mas importante de toda la fiesta, que es definirse
#un modelo de covarianza. En realidad, parece que define un modelo
# completo, en el que se considera, la covarianza, la correlación y 
#el semivariograma. La clave del modelo, esta, en el como se defina
#la correlación.
modelito = gs.Gaussian(dim=2,var=0.5,len_scale=50)

#generamos unos puntos al tuntun para evaluar en ellos la funcion creada
pos = np.array([[-1.,1.],[1.,1.],[1.,-1.],[-1.,-1.]])
Vm = np.zeros(pos.shape[0])
for i in range(pos.shape[0]):
    Vm[i] =  1000*gausianilla(pos[i],sigmar,mu,norm) + \
    2000*gausianilla(pos[i],sigmar1,mu1,norm1) +\
    100*gausianilla(pos[i],sigmar2,mu2,norm2)
Vmt=Vm.copy()
#creanmos el krigenador ;)
krig = gs.krige.Universal(modelito,[pos[:,0],pos[:,1]],Vmt,'linear')
Vhat,var =krig([x,y],mesh_type='structured')

krig.plot() #le dejo pintar a el el campo estimado
#ax.plot_wireframe(xm,ym,Vhat.T,color='r')
plt.figure()
c = plt.contourf(xm,ym,Vhat.T,90)
plt.title('campo vhat')
plt.scatter(pos[:,0],pos[:,1],color='r')
plt.colorbar(c)

maxvh = max(Vmt)
alpha = 0.1
b = 0.2
cs = Vhat+b*var-maxvh
pb = np.array([xm[np.where(cs.T == max(cs.flat))][0],ym[np.where(cs.T == max(cs.flat))][0]])
bnds =((-25,25),(-25,25))

while max(cs.flat) > 0:
   # cos = 
    par =(krig,b,maxvh,'inestructured')
    const = {'type': 'ineq', 'fun': rst, 'args': par }
    ret = pos[-4:]
    #print(ret)
    print(max(cs.flat))
    r = mini(J,pb,(ret[0],ret[1:],alpha),bounds=bnds,constraints=const,options={'disp':True, 'maxiter':1000})
    print(ret[0],r.x)
    vx,varx =krig([r.x[0],r.x[1]])
    print('ext',vx[0]+b*varx[0]-maxvh)
    print(r.success)
    
    if r.success == False:
        r.x = pb
    pos = np.append(pos,np.array([r.x]),0)     
        
    # if r.success == False:
    #     break
   
    #print(pos) 
    Vm =  1000*gausianilla(r.x,sigmar,mu,norm) + \
            2000*gausianilla(r.x,sigmar1,mu1,norm1) +\
            100*gausianilla(r.x,sigmar2,mu2,norm2)
    Vmt= np.append(Vmt,Vm)    
    krig = gs.krige.Universal(modelito,[pos[:,0],pos[:,1]],Vmt,'linear')
    maxvh = max(Vmt)
    Vhatms,varms = krig([x,y],mesh_type='structured')
    cs = Vhatms+b*varms-maxvh
    pb = np.array([xm[np.where(cs.T == max(cs.flat))][0],ym[np.where(cs.T == max(cs.flat))][0]])
#sacamos los resultados en plano que se ven muy bien del resultado de kriging
    if i%8==0:
        
        plt.figure()
        c = plt.contourf(xm,ym,Vhatms.T,30)
        plt.title('campo vhat')
        plt.scatter(pos[:,0],pos[:,1],color='w')
        plt.scatter(pos[-1,0],pos[-1,1],color='r')
        plt.colorbar(c)
        # if r.success == False:
        #     break



# plt.figure()
# plt.contourf(xm,ym,Vhat.T,30)
# #plt.contour(xm,ym,V)
# plt.scatter(pos[:,0],pos[:,1])


plt.figure() 
c = plt.contourf(xm,ym,V,30)
plt.title('campo V')
plt.scatter(pos[:,0],pos[:,1])
plt.colorbar(c)

Vhatms,varms = krig([x,y],mesh_type='structured')

plt.figure()
c=plt.contourf(xm,ym,varms,30)
plt.title('varianza')
plt.colorbar(c)

plt.figure()
c =plt.contourf(xm,ym,Vhatms.T,30)
plt.title('campo vhat')
plt.scatter(pos[:,0],pos[:,1],color='w')
plt.scatter(pos[-1,0],pos[-1,1],color='r')
plt.colorbar(c)