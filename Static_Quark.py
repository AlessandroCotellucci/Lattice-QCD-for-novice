import matplotlib
from matplotlib import pyplot as plt
import math
import cmath
import random
from numpy import *


#Function that compute the dagger of a matrix
def dagger(M):
    N=len(M)
    H=zeros((N,N),dtype=complex)
    R=matrix(M)
    H=R.getH()
    return H.copy()


#Function that generate Nmatrix casual matrix belonging to the SU(3) group
def randommatrixSU3(M):
    identity=eye(3)
    Nmatrix=100
    M=zeros((200,3,3),dtype=complex)
    H=zeros((3,3),dtype=complex)
    eps=0.24
    w=cmath.sqrt(-1)
    for s in range(0,Nmatrix):
        for j in range(0,3):
            for i in range(0,3):
                H[j,i]=complex(random.uniform(-1,1),random.uniform(-1,1))
        H=(H.copy()+dagger(H.copy()))/2. #generation o the hermitian matrice
        for n in range(30): #Taylor series for a SU(3) mstrice
            M[s]=M[s]+(w*eps)**n/math.factorial(n)*linalg.matrix_power(H,n)
        M[s]=M[s]/linalg.det(M[s])**(1/3) #Unitarization

        M[s+Nmatrix]=dagger(M[s]) #Saving the inverse
    return M.copy()


#Function that upload the position using a metropolis algoritm considering the decrising
#in the action using the Wilson action for QCD
#input:-U:array of the link variable
#inner parameters: -N:total number of points in the lattice
def update(U,M):
    Nmatrix=100
    N=8
    beta=5.5
    beta_imp=1.719
    u0=0.797
    improved=False #default use of Wilson action
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                for t in range(0,N):
                    for mi in range(0,4):
                        gamma=Gamma(U,mi,x,y,z,t) #compute Gamma
                        if improved: #condition to use the improved action
                            gamma_imp=Gamma_improved(U,mi,x,y,z,t) #compute Gamma improved if it is asked
                        for p in range(10): #iteration before changing site of the lattice
                            s=random.randint(2,2*Nmatrix) #Choose a random matrix
                            if improved: #condition to use the improved action
                                dS = -beta_imp/(3)*(5/(3*u0**4)*real(trace(dot((dot(M[s],U[x,y,z,t,mi])-U[x,y,z,t,mi]),gamma)))-1/(12*u0**6)*real(trace(dot((dot(M[s],U[x,y,z,t,mi])-U[x,y,z,t,mi]),gamma_imp)))) # change in the improved action
                            else:
                                dS = -beta/(3)*real(trace(dot((dot(M[s].copy(),U[x,y,z,t,mi].copy())-U[x,y,z,t,mi].copy()),gamma.copy()))) # change in action
                            if dS<0 or exp(-dS)>random.uniform(0,1):
                                U[x,y,z,t,mi] = dot(M[s].copy(),U[x,y,z,t,mi].copy())  # update U


#Function that compute gamma for QCD using the Wilson action
#input:-x,y,z,t: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma(U,mi,x,y,z,t):
    N=8
    gamma=0. #inizializing gamma
    incmi=zeros((4),'int')
    incni=zeros((4),'int')
    incmi[mi]=1 #increment in the mi direction
    #next site on mi
    xpmi=(x+incmi[0].copy())%N
    ypmi=(y+incmi[1].copy())%N
    zpmi=(z+incmi[2].copy())%N
    tpmi=(t+incmi[3].copy())%N
    for ni in range(0,4):
        if ni!=mi :
            incni[ni]=1 #increment in the ni direction
            #next site on ni
            xpni=(x+incni[0].copy())%N
            ypni=(y+incni[1].copy())%N
            zpni=(z+incni[2].copy())%N
            tpni=(t+incni[3].copy())%N
            #Previous site on ni
            xmni=(x-incni[0].copy())%N
            ymni=(y-incni[1].copy())%N
            zmni=(z-incni[2].copy())%N
            tmni=(t-incni[3].copy())%N
            #next site on ni and mi
            xpmipni=(x+incmi[0].copy()+incni[0].copy())%N
            ypmipni=(y+incmi[1].copy()+incni[1].copy())%N
            zpmipni=(z+incmi[2].copy()+incni[2].copy())%N
            tpmipni=(t+incmi[3].copy()+incni[3].copy())%N
            #next site on mi and previous on ni
            xpmimni=(x+incmi[0].copy()-incni[0].copy())%N
            ypmimni=(y+incmi[1].copy()-incni[1].copy())%N
            zpmimni=(z+incmi[2].copy()-incni[2].copy())%N
            tpmimni=(t+incmi[3].copy()-incni[3].copy())%N
            incni[ni]=0
            gamma=gamma+dot(dot(U[xpmi,ypmi,zpmi,tpmi,ni],dagger(U[xpni,ypni,zpni,tpni,mi])),dagger(U[x,y,z,t,ni])) \
                        +dot(dot(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,ni]),dagger(U[xmni,ymni,zmni,tmni,mi])),U[xmni,ymni,zmni,tmni,ni])

    return gamma.copy()


#Function that compute gamma for QCD using the improved action using the rctangle terms
#input:-x,y,z,t: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma_improved(U,mi,x,y,z,t):
    N=8
    gamma_imp=0. #inizializing gamma
    incmi=zeros((4),'int')
    incni=zeros((4),'int')
    incmi[mi]=1 #increment in the mi direction
    #next site on mi
    xpmi=(x+incmi[0].copy())%N
    ypmi=(y+incmi[1].copy())%N
    zpmi=(z+incmi[2].copy())%N
    tpmi=(t+incmi[3].copy())%N
    #previous site on mi
    xmmi=(x-incmi[0].copy())%N
    ymmi=(y-incmi[1].copy())%N
    zmmi=(z-incmi[2].copy())%N
    tmmi=(t-incmi[3].copy())%N
    #next site on 2mi
    xp2mi=(x+2*incmi[0].copy())%N
    yp2mi=(y+2*incmi[1].copy())%N
    zp2mi=(z+2*incmi[2].copy())%N
    tp2mi=(t+2*incmi[3].copy())%N

    for ni in range(0,4):
        if ni!=mi :
            incni[ni]=1
            #next site on ni
            xpni=(x+incni[0].copy())%N
            ypni=(y+incni[1].copy())%N
            zpni=(z+incni[2].copy())%N
            tpni=(t+incni[3].copy())%N
            #Previous site on ni
            xmni=(x-incni[0].copy())%N
            ymni=(y-incni[1].copy())%N
            zmni=(z-incni[2].copy())%N
            tmni=(t-incni[3].copy())%N
            #next site on ni and mi
            xpmipni=(x+incmi[0].copy()+incni[0].copy())%N
            ypmipni=(y+incmi[1].copy()+incni[1].copy())%N
            zpmipni=(z+incmi[2].copy()+incni[2].copy())%N
            tpmipni=(t+incmi[3].copy()+incni[3].copy())%N
            #next site on mi and previous on ni
            xpmimni=(x+incmi[0].copy()-incni[0].copy())%N
            ypmimni=(y+incmi[1].copy()-incni[1].copy())%N
            zpmimni=(z+incmi[2].copy()-incni[2].copy())%N
            tpmimni=(t+incmi[3].copy()-incni[3].copy())%N
            #next site on 2ni
            xp2ni=(x+2*incni[0].copy())%N
            yp2ni=(y+2*incni[1].copy())%N
            zp2ni=(z+2*incni[2].copy())%N
            tp2ni=(t+2*incni[3].copy())%N
            #Previous site on 2ni
            xm2ni=(x-2*incni[0].copy())%N
            ym2ni=(y-2*incni[1].copy())%N
            zm2ni=(z-2*incni[2].copy())%N
            tm2ni=(t-2*incni[3].copy())%N
            #next site on 2mi and previous on ni
            xp2mimni=(x+2*incmi[0].copy()-incni[0].copy())%N
            yp2mimni=(y+2*incmi[1].copy()-incni[1].copy())%N
            zp2mimni=(z+2*incmi[2].copy()-incni[2].copy())%N
            tp2mimni=(t+2*incmi[3].copy()-incni[3].copy())%N
            #next site on mi and previous on 2ni
            xpmim2ni=(x+incmi[0].copy()-2*incni[0].copy())%N
            ypmim2ni=(y+incmi[1].copy()-2*incni[1].copy())%N
            zpmim2ni=(z+incmi[2].copy()-2*incni[2].copy())%N
            tpmim2ni=(t+incmi[3].copy()-2*incni[3].copy())%N
            #previous site on ni and mi
            xmmimni=(x-incmi[0].copy()-incni[0].copy())%N
            ymmimni=(y-incmi[1].copy()-incni[1].copy())%N
            zmmimni=(z-incmi[2].copy()-incni[2].copy())%N
            tmmimni=(t-incmi[3].copy()-incni[3].copy())%N
            #next site on ni and previous mi
            xmmipni=(x-incmi[0].copy()+incni[0].copy())%N
            ymmipni=(y-incmi[1].copy()+incni[1].copy())%N
            zmmipni=(z-incmi[2].copy()+incni[2].copy())%N
            tmmipni=(t-incmi[3].copy()+incni[3].copy())%N
            incni[ni]=0

            gamma_imp=gamma_imp+dot(dot(U[xpmi,ypmi,zpmi,tpmi,mi],U[xp2mi,yp2mi,zp2mi,tp2mi,ni]),dot(dagger(U[xpmipni,ypmipni,zpmipni,tpmipni,mi]),dot(dagger(U[xpni,ypni,zpni,tpni,mi]),dagger(U[x,y,z,t,ni])))) \
                                +dot(dot(U[xpmi,ypmi,zpmi,tpmi,mi],dagger(U[xp2mimni,yp2mimni,zp2mimni,tp2mimni,ni])),dot(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,mi]),dot(dagger(U[xmni,ymni,zmni,tmni,mi]),U[xmni,ymni,zmni,tmni,ni]))) \
                                +dot(dot(U[xpmi,ypmi,zpmi,tpmi,ni],U[xpmipni,ypmipni,zpmipni,tpmipni,ni]),dot(dagger(U[xp2ni,yp2ni,zp2ni,tp2ni,mi]),dot(dagger(U[xpni,ypni,zpni,tpni,ni]),dagger(U[x,y,z,t,ni])))) \
                                +dot(dot(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,ni]),dagger(U[xpmim2ni,ypmim2ni,zpmim2ni,tpmim2ni,ni])),dot(dagger(U[xm2ni,ym2ni,zm2ni,tm2ni,mi]),dot(U[xm2ni,ym2ni,zm2ni,tm2ni,ni],U[xmni,ymni,zmni,tmni,ni]))) \
                                +dot(dot(U[xpmi,ypmi,zpmi,tpmi,ni],dagger(U[xpni,ypni,zpni,tpni,mi])),dot(dagger(U[xmmipni,ymmipni,zmmipni,tmmipni,mi]),dot(dagger(U[xmmi,ymmi,zmmi,tmmi,ni]),U[xmmi,ymmi,zmmi,tmmi,mi]))) \
                                +dot(dot(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,ni]),dagger(U[xmni,ymni,zmni,tmni,mi])),dot(dagger(U[xmmimni,ymmimni,zmmimni,tmmimni,mi]),dot(U[xmmimni,ymmimni,zmmimni,tmmimni,ni],U[xmmi,ymmi,zmmi,tmmi,mi])))
    return gamma_imp.copy()


#Function that compute the side of a Wilson loop
#input:-x,y,z,t: position in which gamma is computed
#      -U:array of link variables
#      -f:direction
#inner parameter:-N:total number of points in the lattice
def ProductU(U,x,y,z,t,n,f):
    N=8
    I=eye(3)
    productU=zeros((3,3),dtype=complex) #allocation of productU
    productU=I.copy()  #inizializing ProductU
    inc=zeros((4),'int')
    inc[f]=1
    for i in range(n):
        x_inc=(x+i*inc[0])%N #increment in the Wilson line direction
        y_inc=(y+i*inc[1])%N #increment in the Wilson line direction
        z_inc=(z+i*inc[2])%N #increment in the Wilson line direction
        t_inc=(t+i*inc[3])%N #increment in the Wilson line direction
        productU=dot(productU,U[x_inc,y_inc,z_inc,t_inc,f])
    return productU.copy()

#Function that compute the inverse side of a Wilson loop
#input:-x,y,z,t: position in which gamma is computed
#      -U:array of link variables
#      -f:direction
#inner parameter:-N:total number of points in the lattice
def ProductUdagger(U,x,y,z,t,n,f):
    N=8
    I=eye(3)
    productUdagger=zeros((3,3),dtype=complex) #allocation of productUdagger
    productUdagger=I.copy() #inizializing ProductUdagger
    inc=zeros((4),'int')
    inc[f]=1
    for i in range(n):
        x_inc=(x-(i+1)*inc[0])%N #decrement in the Wilson line direction
        y_inc=(y-(i+1)*inc[1])%N #decrement in the Wilson line direction
        z_inc=(z-(i+1)*inc[2])%N #decrement in the Wilson line direction
        t_inc=(t-(i+1)*inc[3])%N #decrement in the Wilson line direction
        productUdagger=dot(productUdagger,dagger(U[x_inc,y_inc,z_inc,t_inc,f]))
    return productUdagger.copy()

#Function that compute the Wilson Loop time*axradius*a mean value on the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -time, radius:dimension of the loop
#inner parameters:-N:points in the lattice
def compute_WL(U,time,radius):
    N=8
    productUt=zeros((3,3),dtype=complex) #allocation
    productUr=zeros((3,3),dtype=complex) #allocation
    productUtdagger=zeros((3,3),dtype=complex) #allocation
    productUrdagger=zeros((3,3),dtype=complex) #allocation
    WL=0.
    for y in range(N):
        for z in range(N):
            for t in range(N):
                for x in range(N):
                    productUt=ProductU(U,x,y,z,t,time,3)
                    productUr=ProductU(U,x,y,z,(t+time)%N,radius,0)
                    productUtdagger=ProductUdagger(U,(x+radius)%N,y,z,(t+time)%N,time,3)
                    productUrdagger=ProductUdagger(U,(x+radius)%N,y,z,t,radius,0)
                    WL=WL+real(trace(dot(dot(productUt,productUr),dot(productUtdagger,productUrdagger))))

    return WL/3./N**4




#allocation of the arrays and definition of the parameters
a=0.25
N=8
N_sim=5
Nmatrix=200
N_cf=20
N_cor=50
U=zeros((N,N,N,N,4,3,3),dtype=complex) #inizializing U
WL=ones((N_cf,N_sim,N_sim), 'double') #inizializing WL
M=zeros((Nmatrix,3,3),dtype=complex) #inizializing the random matrix
avg_WL=zeros((N_sim,N_sim),'double') #inizializing avg_WL
avg_WLSQ=zeros((N_sim,N_sim),'double') #inizializing avg_WLSQ
err_avg_WL=zeros((N_sim,N_sim),'double') #inizializing err_avg_WL
potential=zeros(N_sim-1) #inizializing the potential
err_potential=zeros(N_sim-1) #inizializing the err_potential
rad=zeros(N_sim-1) #inizializing the radius
# inizializing U with the identity which belongs to the SU(3) group
for x in range(0,N):
    for y in range(0,N):
        for z in range(0,N):
            for t in range(0,N):
                for mi in range(0,4):
                    for n in range(0,3):
                        U[x,y,z,t,mi,n,n]=1.
# Generation of the random matrix
M=randommatrixSU3(M)
#Computation of the Monte Carlo mean value for the propagator in every time
#in the lattice
for j in range(0,2*N_cor): # thermalize U
    update(U,M)
print('Termalized')
for alpha in range(0,N_cf): # loop on random paths
    for j in range(0,N_cor):
        update(U,M)
    WL[alpha]=0.
    for time in range(1,N_sim):
        for radius in range(1,N_sim):
            WL[alpha,time,radius]=compute_WL(U,time,radius)
    print("End simulation number:",alpha+1)
    print(WL[alpha])
for time in range(1,N_sim): #computation of the Wilson loop mean value for every possible values of time and radius
    for radius in range(1,N_sim):
        avg_WL[time,radius]=0.
        avg_WLSQ[time,radius]=0.
        for alpha in range(0,N_cf): # compute MC averages
            avg_WL[time,radius]= avg_WL[time,radius]+WL[alpha,time,radius]
            avg_WLSQ[time,radius]=avg_WLSQ[time,radius]+WL[alpha,time,radius]**2
        avg_WL[time,radius] = avg_WL[time,radius]/N_cf  #mean value on every configuration
        avg_WLSQ[time,radius] = avg_WLSQ[time,radius]/N_cf
        err_avg_WL[time,radius]=((avg_WLSQ[time,radius]-avg_WL[time,radius]**2)/N_cf)**(1/2) #statistical error
print(avg_WL,err_avg_WL)
for i in range(0,N_sim-1): #high time limit
    radius=i+1
    potential[i]=avg_WL[N_sim-2,radius]/avg_WL[N_sim-1,radius] #computation of the potential
    err_potential[i]=((err_avg_WL[N_sim-2,radius]/avg_WL[N_sim-2,radius])**2+(err_avg_WL[N_sim-1,radius]/avg_WL[N_sim-1,radius])**2)**(1/2)*potential[i] #propagation of the error on the potential
    rad[i]=radius

print(potential, err_potential)
#plot of the numerical solution
plt.errorbar(rad, potential, yerr=err_potential, fmt='.', color='black', label='Numerical');
plt.legend(loc='upper right')
plt.title('Static quark potential')
plt.xlabel('$r/a$')
plt.ylabel('$V(r)a$')
plt.show()

