# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:44:04 2025
Playing with kriging and other fancy stuff
Calculado con los parametros del artículo para ver qué sale
kriginng_util tiene una función para generar gausianas, centro y varianza a 
definir y también con la posibilidad de rotarlas
 las funciones auxiliares para definir J (la funcion
que s equiere optimizar y rest (la restriccion). Ver el artículo de Athur Kanh
hal-01170131v2)
 
En este caso se ha optimizado empleando direct. Este metodo busca el mínimo
global en un entorno acotado (límites a las coordenadas del espacio de busqueda)
Para unir la función a optimizar con las restricciones se ha creado una única
función Jrst. El truco es que si no se cumple la restricción, a la función de
coste se le añade un valor 10^6. Esto hace que el mínimo caiga fuera de los
límites que se exploran. por tanto direct desechará la regiones donde pase esto
y solo buscará donde la restricción se cumpla ( a al menos, eso creo)
@author: abierto
"""
from kriging_util import *
from scipy.optimize import direct
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
((ax.plot_wireframe(xm,ym,V)))
plt.xlabel('x')


#ahora viene lo mas importante de toda la fiesta, que es definirse
#un modelo de covarianza. En realidad, parece que define un modelo
# completo, en el que se considera, la covarianza, la correlación y 
#el semivariograma. La clave del modelo, esta, en el como se defina
#la correlación.
modelito = gs.Gaussian(dim=2,var=0.5,len_scale=50)

#generamos unos puntos al tuntun para evaluar en ellos la funcion creada
#pos = np.array([[-1.,1.],[1.,1.],[1.,-1.],[-1.,-1.]])
#pos = np.array([[-10.,8.],[1.,1.],[1.,-1.],[-1.,-1.]])
pos = np.array([[-1.,1.],[1.,1.],[10.,-4.],[-1.,-1.]])
#mido el campo en los puntos que tengo
Vm = np.zeros(pos.shape[0])
for i in range(pos.shape[0]):
    Vm[i] =  1000*gausianilla(pos[i],sigmar,mu,norm) + \
    2000*gausianilla(pos[i],sigmar1,mu1,norm1) +\
    100*gausianilla(pos[i],sigmar2,mu2,norm2)
#Vmyt va a contener todas la medidas que se realizen a lo largo de la búqueda
Vmt=Vm.copy()

#creanmos el krigenador con los puntos medidos ;)
krig = gs.krige.Universal(modelito,[pos[:,0],pos[:,1]],Vmt,'linear')

#y optenemos estima de campo y varianza sobro todos los puntos de nuestra 
#cuadrícula de trabajo 
Vhat,var =krig([x,y],mesh_type='structured')

krig.plot() #le dejo pintar a el el campo estimado
#ax.plot_wireframe(xm,ym,Vhat.T,color='r')
plt.figure()
#por como creo los valores del campo con bucles en x e y, la matriz
#que optengo es la traspuesta a la real XP. Todo:arreglar la chapuza.
#lo pinto para comparar con el que pinta krig y ver que lo que digo es cierto
c = plt.contourf(xm,ym,Vhat.T,90)
plt.title('campo vhat')

#pinto sogre e campo los puntos iniciales
plt.scatter(pos[:,0],pos[:,1],color='r')
plt.colorbar(c)

#calculo el valor máximo medido so far y la posición
maxvh = max(Vmt)
pmax = pos[np.where(Vmt.flat == max(Vmt))]

#####Parámetros del modelo de búsqueda####################
alpha = 0.25 #peso de la suma de la distancia del punto a los vecinos 
b = 10 #peso que se da a la varianza en la función de restricción

#valor de la restricción sobre todos los puntos de la cuadrícula
cs = Vhat+b*var**0.5-maxvh

bnds =((-25,25),(-25,25)) #cotas en x e y del D, el espacio de búsqueda
i = 0 #iniciamos unn contador para saber cuantas iteraciones está haciendo

###############Inicio Bucle de Búsqueda########################################
while (max(cs.flat) > 0)&(i < 1000):
    #vamos a calcular mientras se cumpla la condición en los puntos de la
    #cuadricula, de que al menos hay uno en que se cumple la restricción. 
    #Se podría apretar más pero creo que no tiene sentido
    #El número de iteraciones máximo es para proteger de bucles infinitos
    #Si se ha evaluado el campo en mil puntos el algorithmo es muuuy malo
    
    ret = pos[-4:] #últimas posiciones de los cuatro agentes (TODO: hacer este numero adaptable)
    
    ############optimizador####################################################
    # maxfun está de hecho en su valor por defecto = 1000n (n dimension del es
    #pacio de busqueda) (numero maximo de evaluaciones de la funcion objetivo)
    #la len_tol define el tamño de cuadricula (lado) por debajo del cual se considera
    #que el algoritmo ha convergido por defecto es 1e-6. Si se baja el valor
    # en algunos pasos no converge porque agota maxfun
    #ver ayuda de scipy. es posible que se pueda ajustar mejor
    #muy importante siempre se calcula la nueva posición para el cuarto
    #agente empezando a contar or la cola ret[0]. De este modo, cada cuatro
    #iteraciones se han renovado las posiciones de los cuatro agentes.
    
    r = direct(Jrst,bnds,args =(ret[0],ret[1:],alpha,krig,b,maxvh),maxfun = 2000,len_tol=1e-3)
    ###########################################################################
    
    #evaluamos la estima de campo y varianza en el punto encontrado
    vx,varx =krig([r.x[0],r.x[1]])
    #imprimimos info de lo que está pasando. Se puede omitir para que 
    #calcule más deprisa
    print(f'i={i}\nmax={max(cs.flat)}\nrest={vx[0]+b*varx[0]-maxvh}\n{r}')
        
    #añadimos el punto a las posiciones visitadas
    pos = np.append(pos,np.array([r.x]),0)     
        
    #medimos en el punto nuevo 
    Vm =  1000*gausianilla(r.x,sigmar,mu,norm) + \
            2000*gausianilla(r.x,sigmar1,mu1,norm1) +\
            100*gausianilla(r.x,sigmar2,mu2,norm2)
    #si la nueva medida es el valor mas grande obtenido so far, lo guardamos
    #Así como la posicion        
    if Vm > maxvh:
        maxvh = Vm
        pmax = r.x #aqui se guarda la posición del máximo
    #Añadimos la medida a las que ya tenemos
    Vmt= np.append(Vmt,Vm)
    #y calculamos un nuevo modelo para kriging, añadiendo la nueva posicion
    #y la nueva medida    
    krig = gs.krige.Universal(modelito,[pos[:,0],pos[:,1]],Vmt,'linear')
    
    #usamos el modelo para volver a estimar en la cuadricula
    Vhatms,varms = krig([x,y],mesh_type='structured')
    #y así obtener una nueva condición de parada, que al menos haya un punto 
    # de la restricción por encima de cero
    cs = Vhatms+b*varms**0.5-maxvh
    
    #de vez en cuando (ahora cada cinco iteraciones) dibujamos los resultados
    #permite ver lo que pasa con los campos estimados pero consume recursos...
    #y tiempo. posiblemente lo menor es comentarlo, o diezmar las figuras
    #obtenidas si necesita muchas iteraciones
    if i%5==0:
        
        plt.figure()
        c = plt.contourf(xm,ym,Vhatms.T,30)
        plt.title('campo vhat')
        plt.scatter(pos[:,0],pos[:,1],color='w')
        plt.scatter(pos[-1,0],pos[-1,1],color='r')
        plt.colorbar(c)
        
    #incrementamos el contador
    i += 1
############## Fin del bucle de busqueda ######################################

###############forensic report#################################################

#pintamos el campo real
plt.figure() 
c = plt.contourf(xm,ym,V,30) #campo real
plt.title('campo V') 
plt.scatter(pos[0::4,0],pos[0::4,1],color = 'w') #puntos recorridos por todos los agentes
plt.scatter(pos[1::4,0],pos[1::4,1],color = 'b')
plt.scatter(pos[2::4,0],pos[2::4,1],color = 'r')
plt.scatter(pos[3::4,0],pos[3::4,1],color = 'g')
plt.scatter(pmax[0],pmax[1],color='m',marker ='*',s = 10) #maximo encontrado
plt.colorbar(c)

#este calculo sobra es el mimo que el de la linea 156
Vhatms,varms = krig([x,y],mesh_type='structured')
#Eliminamos la parte negatica del campo estimado (la hacemos cero). La idea es resaltar la parte
#de la estima que tiene sentido, nuestro campo gaussiano no tiene valores
#negativos. Si los tuviera esto no debería hacerse ¿?
Vhatfinal = Vhatms.copy()
for i in range(Vhatfinal.shape[0]):
    for j in range(Vhatfinal.shape[1]):
        if Vhatfinal[i,j] < 0:
            Vhatfinal[i,j] = 0 

#pintamos la varianza final
plt.figure()
c=plt.contourf(xm,ym,varms.T,30)
plt.title('varianza')
plt.colorbar(c)

#pintamos el campo estimado
plt.figure()
c =plt.contourf(xm,ym,Vhatfinal.T,30) #campo estimado
plt.title('campo vhat')
plt.scatter(pos[:,0],pos[:,1],color='w') #puntos visitados
plt.scatter(pos[-1,0],pos[-1,1],color='r') #último punto medido
plt.scatter(pmax[0],pmax[1],color='g',marker ='*') #posición del máximo encontrado
plt.colorbar(c)
#Iteración en que se encontro el maximo (ojo en realidad es medida en que se
#encontró)

print(np.where(Vmt == maxvh))
