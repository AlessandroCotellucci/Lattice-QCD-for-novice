import matplotlib
from matplotlib import pyplot as plt
import vegas
import math
import random
from numpy import *

#Function that upload the position using a metropolis algoritm considering the decrising
#in the action
#input:-x:vector of the position
#inner parameter:-eps: intevar in which is randomly picked the perturbation
#                -N:total number of points in the lattice
def update(x):
    N=20
    eps=1.4
    for j in range(0,N):
        old_x = x[j] # save original value
        old_Sj = S(j,x)
        x[j] = x[j] + random.uniform(-eps,eps) # update x[j]
        dS = S(j,x) - old_Sj # change in action
        if dS>0 and exp(-dS)<random.uniform(0,1):
                x[j] = old_x # restore old value

#Function that compute the action of an harmonic oscillator in 1D in the j-ism
#poin of the vector x
#input:-j:position in which the action is computed
#      -x:vector of the position
#inner parameter:-a:discretization of the position
#                -N:total number of points in the lattice
def S(j,x): # harm. osc. S
    N=20
    a=1/2
    jp = (j+1)%N # next site
    jm = (j-1)%N # previous site
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

def compute_G(x,n):
    g = 0
    N=20
    for j in range(0,N):
        g = g + (x[j]**3)*(x[(j+n)%N]**3)
    return g/N
#allocation of the arrays and definition of the parameters
a=1/2
N=20
N_cf=1000
N_cor=20
x=ones(N, 'double')
G=ones((N_cf,N), 'double')
avg_G=ones(N, 'double')
avg_Gsquare=ones(N, 'double')
errorG=ones(N, 'double')

#Computation of the Monte Carlo mean value for the propagator in every time
#in the lattice
for j in range(0,N): # initialize x
    x[j] = 0
for j in range(0,5*N_cor): # thermalize x
    update(x)
for alpha in range(0,N_cf): # loop on random paths
    for j in range(0,N_cor):
        update(x)
    for n in range(0,N):
        G[alpha][n] = compute_G(x,n)
for n in range(0,N): # compute MC averages
    avg_G[n] = 0
    avg_Gsquare[n] = 0
    for alpha in range(0,N_cf):
        avg_G[n] = avg_G[n] + G[alpha][n]
        avg_Gsquare[n]=avg_Gsquare[n] + (G[alpha][n])**2
    avg_G[n] = avg_G[n]/N_cf  #mean value on every path of the propagator
    avg_Gsquare[n]=avg_Gsquare[n]/N_cf
    errorG[n]=((avg_Gsquare[n]-(avg_G[n])**2)/N_cf)**(1/2)
    print(n*a,avg_G[n],errorG[n]) #print of the result for every point n

#Allocation of the arrays related to the variation in energy
errorE=ones(N-1, 'double')
deltaE=ones(N-1, 'double')
t=ones(N-1, 'double')
exact=ones(N-1, 'double')

#Computation of the variation in energy for every point of the lattice
for n in range(0,N-1):
    deltaE[n]=log(abs(avg_G[n]/avg_G[n+1]))/a
    errorE[n]=(((errorG[n]*avg_G[n+1])/(a*avg_G[n]))**2+((errorG[n+1]*avg_G[n])/(a*avg_G[n+1]))**2)**(1/2)
    t[n]=n*a
    exact[n]=1
    print(n*a,deltaE[n],errorE[n])

#Plot of the exact and the numerical solution
plt.plot(t,exact,'b',label='Exact')
plt.errorbar(t, deltaE, yerr=errorE, fmt='.', color='black', label='Numerical');
plt.axis([0,3,0,2])
plt.legend(loc='upper right')
plt.title('Harmonic oscillator')
plt.xlabel('t')
plt.ylabel(r'$\Delta E(t)$')
plt.show()
