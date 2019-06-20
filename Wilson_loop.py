import matplotlib
from matplotlib import pyplot as plt
import vegas
import math
import cmath
import random
from numpy import *
#Function that compute the trace of a matrix
def trace(M):
    N=len(M)
    trace=0.
    for i in range(0,N):
        trace=trace+M[i,i]
    return trace





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
        H=(H.copy()+dagger(H.copy()))/2.
        for n in range(30):
            M[s]=M[s]+(w*eps)**n/math.factorial(n)*linalg.matrix_power(H,n)
        M[s]=M[s]/linalg.det(M[s])**(1/3)

        M[s+Nmatrix]=dagger(M[s])
    return M.copy()

#Function that compute the dagger of a matrix
def dagger(M):
    N=len(M)
    H=zeros((N,N),dtype=complex)
    for i in range(0,N):
        for j in range(0,N):
            H[i,j]=conjugate(M[j,i].copy())
    return H.copy()

#Function that compute the product of two square matrix with equal dimension
def rXc(M,H):
    N=len(M)
    R=zeros((N,N),dtype=complex)
    for j in range(0,N):
        for i in range(0,N):
            for n in range(0,N):
                R[i,j]=R[i,j]+M[i,n].copy()*H[n,j].copy()
    return R.copy()



#Function that upload the position using a metropolis algoritm considering the decrising
#in the action using the easiest action for QCD
#input:-U:array of the link variable
#inner parameters: -N:total number of points in the lattice
def update(U,M):
    Nmatrix=100
    N=4
    beta=5.5
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                for t in range(0,N):
                    for mi in range(0,4):
                        old_U = U[x,y,z,t,mi].copy() # save original value
                        s=random.randint(2,2*Nmatrix) #Choose a random matrix
                        gamma=Gamma(U,mi,x,y,z,t) #compute Gamma

                        U[x,y,z,t,mi] = rXc(M[s].copy(),U[x,y,z,t,mi].copy()) # update U
                        #deter=linalg.det(U[t,x,y,z,mi])
                        #err=abs(1.-deter)
                        #print(trace(U[t,x,y,z,mi]))
                        #if err>1:
                        #    print(deter,linalg.det(M[s]),linalg.det(old_U),s)
                        #    print(U[t,x,y,z,mi],M[s],old_U)
                        dS = -beta/(3)*real(trace(rXc((U[x,y,z,t,mi].copy()-old_U.copy()),gamma.copy()))) # change in action
                        if dS>0 and exp(-dS)<random.uniform(0,1):
                            U[x,y,z,t,mi] = old_U.copy() # restore old value


#Function that compute gamma for QCD using the easiest action
#input:-i,j,k,l: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma(U,mi,x,y,z,t):
    N=4
    gamma=0.

    for ni in range(0,4):
        #upload of the poin on mi
        if mi==3:
                #next site on mi
                tpmi=(t+1)%N
                xpmi=x
                ypmi=y
                zpmi=z
                #next site on ni and mi
                tpmipni=(t+1)%N
                xpmipni=x
                ypmipni=y
                zpmipni=z
                #next site on mi and previous on ni
                tpmimni=(t+1)%N
                xpmimni=x
                ypmimni=y
                zpmimni=z
        if mi==0:
                #next site on mi
                tpmi=t
                xpmi=(x+1)%N
                ypmi=y
                zpmi=z
                #next site on ni and mi
                tpmipni=t
                xpmipni=(x+1)%N
                ypmipni=y
                zpmipni=z
                #next site on mi and previous on ni
                tpmimni=t
                xpmimni=(x+1)%N
                ypmimni=y
                zpmimni=z
        if mi==1:
                #next site on mi
                tpmi=t
                xpmi=x
                ypmi=(y+1)%N
                zpmi=z
                #next site on ni and mi
                tpmipni=t
                xpmipni=x
                ypmipni=(y+1)%N
                zpmipni=z
                #next site on mi and previous on ni
                tpmimni=t
                xpmimni=x
                ypmimni=(y+1)%N
                zpmimni=z
        if mi==2:
                #next site on mi
                tpmi=t
                xpmi=x
                ypmi=y
                zpmi=(z+1)%N
                #next site on ni and mi
                tpmipni=t
                xpmipni=x
                ypmipni=y
                zpmipni=(z+1)%N
                #next site on mi and previous on ni
                tpmimni=t
                xpmimni=x
                ypmimni=y
                zpmimni=(z+1)%N
        if ni!=mi :
            #unpload of the poin on ni
            if ni==3:
                #next site on ni
                tpni=(t+1)%N
                xpni=x
                ypni=y
                zpni=z
                #Previous site on ni
                tmni=(t-1)%N
                xmni=x
                ymni=y
                zmni=z
                #next site on ni and mi
                tpmipni=(tpmipni+1)%N
                xpmipni=xpmipni
                ypmipni=ypmipni
                zpmipni=zpmipni
                #next site on mi and previous on ni
                tpmimni=(tpmimni-1)%N
                xpmimni=xpmimni
                ypmimni=ypmimni
                zpmimni=zpmimni
            if ni==0:
                #next site on ni
                tpni=t
                xpni=(x+1)%N
                ypni=y
                zpni=z
                #Previous site on ni
                tmni=t
                xmni=(x-1)%N
                ymni=y
                zmni=z
                #next site on ni and mi
                tpmipni=tpmipni
                xpmipni=(xpmipni+1)%N
                ypmipni=ypmipni
                zpmipni=zpmipni
                #next site on mi and previous on ni
                tpmimni=tpmimni
                xpmimni=(xpmimni-1)%N
                ypmimni=ypmimni
                zpmimni=zpmimni
            if ni==1:
                #next site on ni
                tpni=t
                xpni=x
                ypni=(y+1)%N
                zpni=z
                #Previous site on ni
                tmni=t
                xmni=x
                ymni=(y-1)%N
                zmni=z
                #next site on ni and mi
                tpmipni=tpmipni
                xpmipni=xpmipni
                ypmipni=(ypmipni+1)%N
                zpmipni=zpmipni
                #next site on mi and previous on ni
                tpmimni=tpmimni
                xpmimni=xpmimni
                ypmimni=(ypmimni-1)%N
                zpmimni=zpmimni
            if ni==2:
                #next site on ni
                tpni=t
                xpni=x
                ypni=y
                zpni=(z+1)%N
                #previous site on ni
                tmni=t
                xmni=x
                ymni=y
                zmni=(z-1)%N
                #next site on ni and mi
                tpmipni=tpmipni
                xpmipni=xpmipni
                ypmipni=ypmipni
                zpmipni=(zpmipni+1)%N
                #next site on mi and previous on ni
                tpmimni=tpmimni
                xpmimni=xpmimni
                ypmimni=ypmimni
                zpmimni=(zpmimni-1)%N

            gamma=gamma+rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,ni].copy(),dagger(U[xpni,ypni,zpni,tpni,mi].copy())),dagger(U[x,y,z,t,ni].copy()))
            gamma=gamma+rXc(rXc(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,ni].copy()),dagger(U[xmni,ymni,zmni,tmni,mi].copy())),U[xmni,ymni,zmni,tmni,ni].copy())

    return gamma.copy()

#Function that compute the Wilson Loop for each point of the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -i,j,k,l:position computed
#inner parameters:-N:points in the lattice
def compute_WL(U,t,x,y,z):
    N=4
    WL = 0.
    for mi in range(0,4):
        #upload of the poin on mi
        if mi==3:
                #next site on mi
                tpmi=(t+1)%N
                xpmi=x
                ypmi=y
                zpmi=z
                #next site on ni and mi
                tpmipni=(t+1)%N
                xpmipni=x
                ypmipni=y
                zpmipni=z
        if mi==0:
                #next site on mi
                tpmi=t
                xpmi=(x+1)%N
                ypmi=y
                zpmi=z
                #next site on ni and mi
                tpmipni=t
                xpmipni=(x+1)%N
                ypmipni=y
                zpmipni=z
        if mi==1:
                #next site on mi
                tpmi=t
                xpmi=x
                ypmi=(y+1)%N
                zpmi=z
                #next site on ni and mi
                tpmipni=t
                xpmipni=x
                ypmipni=(y+1)%N
                zpmipni=z
        if mi==2:
                #next site on mi
                tpmi=t
                xpmi=x
                ypmi=y
                zpmi=(z+1)%N
                #next site on ni and mi
                tpmipni=t
                xpmipni=x
                ypmipni=y
                zpmipni=(z+1)%N
        for ni in range(0,mi):
            if ni!=mi :
                #unpload of the poin on ni
                if ni==3:
                    #next site on ni
                    tpni=(t+1)%N
                    xpni=x
                    ypni=y
                    zpni=z
                    #next site on ni and mi
                    tpmipni=(tpmipni+1)%N
                    xpmipni=xpmipni
                    ypmipni=ypmipni
                    zpmipni=zpmipni
                if ni==0:
                    #next site on ni
                    tpni=t
                    xpni=(x+1)%N
                    ypni=y
                    zpni=z
                    #next site on ni and mi
                    tpmipni=tpmipni
                    xpmipni=(xpmipni+1)%N
                    ypmipni=ypmipni
                    zpmipni=zpmipni
                if ni==1:
                    #next site on ni
                    tpni=t
                    xpni=x
                    ypni=(y+1)%N
                    zpni=z
                    #next site on ni and mi
                    tpmipni=tpmipni
                    xpmipni=xpmipni
                    ypmipni=(ypmipni+1)%N
                    zpmipni=zpmipni
                if ni==2:
                    #next site on ni
                    tpni=t
                    xpni=x
                    ypni=y
                    zpni=(z+1)%N
                    #next site on ni and mi
                    tpmipni=tpmipni
                    xpmipni=xpmipni
                    ypmipni=ypmipni
                    zpmipni=(zpmipni+1)%N
                WL=WL+trace(rXc(U[x,y,z,t,mi].copy(),rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,ni].copy(),dagger(U[xpni,ypni,zpni,tpni,mi].copy())),dagger(U[x,y,z,t,ni].copy()))))

    return real(WL)/(3.*12)

#allocation of the arrays and definition of the parameters
a=0.25
N=4
Nmatrix=200
N_cf=10
N_cor=20
U=zeros((N,N,N,N,4,3,3),dtype=complex) # inizializing U
WL=ones((N_cf), 'double')
M=zeros((Nmatrix,3,3),dtype=complex) #inizializing the random matrix
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
for j in range(0,5*N_cor): # thermalize U
    update(U,M)
print('Termalized')
for alpha in range(0,N_cf): # loop on random paths
    for j in range(0,N_cor):
        update(U,M)
    WL[alpha]=0.
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                for t in range(0,N):
                    WL[alpha] =WL[alpha]+compute_WL(U,t,x,y,z)

    WL[alpha]=WL[alpha]/N**4
    print(WL[alpha])
avg_WL = 0.
for alpha in range(0,N_cf): # compute MC averages
    avg_WL= avg_WL+WL[alpha]
avg_WL = avg_WL/N_cf  #mean value on every configuration
print(avg_WL) #print of the result
