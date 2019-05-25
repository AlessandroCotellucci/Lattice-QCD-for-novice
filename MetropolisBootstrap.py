import matplotlib
from matplotlib import pyplot as plt
import vegas
import math
import random
from numpy import *

#Function that randomly rearrange the simulation to gives more solutions using the
#'bootstrap' method
#input:-G:solution of the Monte Carlo simulation
#inner parameters:-N:number of points in the lattice
def bootstrap(G):
    N=len(x)
    N_cf = len(G)
    G_bootstrap=ones((N_cf,N), 'double')     # new ensemble
    for i in range(0,N_cf):
        alpha = int(random.uniform(0,N_cf)) # choose random config
        G_bootstrap[i]=G[alpha] # keep G[alpha]
    return G_bootstrap

#Function that upload the position using a metropolis algoritm considering the decrising
#in the action
#input:-x:vector of the position
#inner parameter:-eps: intevar in which is randomly picked the perturbation
#                -N:total number of points in the lattice
def update(x):
    N=len(x)
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
    N=len(x)
    a=1/2
    jp = (j+1)%N # next site
    jm = (j-1)%N # previous site
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

#Function that compute the function computed as mean value on the path integral
#input:-x:vector of the positions
#      -n:position computed
#inner parameters:-N:points in the lattice
def compute_G(x,n):
    g = 0
    N=len(x)
    for j in range(0,N):
        g = g + x[j]*x[(j+n)%N]
    return g/N

#allocation of the arrays and definition of the parameters
a=1/2
N=20
N_cf=1000
N_cor=20
N_stat=100
x=ones(N, 'double')
G=ones((N_cf,N), 'double')
avg_G=ones((N,N_stat), 'double')
simulations_G=ones((N_stat,N_cf,N), 'double')

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

simulations_G[0]=G #Keep as 0 simulation the result of the monte carlo simulation

for k in range(1,N_stat):
    simulations_G[k]=bootstrap(G) #Inizialization of the N_stat-1 simulations

for k in range(0,N_stat):
    for n in range(0,N): # compute of the average on all the simulations
        avg_G[n][k] = 0
        for alpha in range(0,N_cf):
            avg_G[n][k] = avg_G[n][k] + simulations_G[k][alpha][n]
        avg_G[n][k] = avg_G[n][k]/N_cf  #mean value on every path of the propagator
        #print(n*a,avg_G[n][k]) #print of the result for every point n




#Allocation of the arrays related to the variation in energy
deltaE=ones((N-1,N_stat), 'double')
mean_deltaE=ones(N-1, 'double')
mean_deltaE_square=ones(N-1, 'double')
error_deltaE=ones(N-1, 'double')
t=ones(N-1, 'double')
exact=ones(N-1, 'double')

#Computation of the variation in energy for every point of the lattice
for k in range(0,N_stat):
    for n in range(0,N-1):
        deltaE[n][k]=log(abs(avg_G[n][k]/avg_G[n+1][k]))/a
        t[n]=n*a
        exact[n]=1
        #print(n*a,deltaE[n][k])

#Statistical analisis on the variation in energy
for n in range(0,N-1):
    mean_deltaE[n]=0.
    mean_deltaE_square[n]=0.
    for k in range(0,N_stat): #computation of mean value of delta energy and the standard deviation
        mean_deltaE[n]=mean_deltaE[n]+deltaE[n][k]
        mean_deltaE_square[n]=mean_deltaE_square[n]+(deltaE[n][k])**2
    mean_deltaE[n]=mean_deltaE[n]/N_stat
    mean_deltaE_square[n]=mean_deltaE_square[n]/N_stat
    error_deltaE[n]=(mean_deltaE_square[n]-(mean_deltaE[n])**2)**(1/2)
    print(n*a,mean_deltaE[n],error_deltaE[n])

#Plot of the exact and the numerical solution
plt.plot(t,exact,'b',label='Exact')
plt.errorbar(t, mean_deltaE, yerr=error_deltaE, fmt='.', color='black', label='Numerical');
#plt.axis([0,3,0,2])
plt.legend(loc='upper right')
plt.title('Harmonic oscillator')
plt.xlabel('t')
plt.ylabel(r'$\Delta E(t)$')
plt.show()
