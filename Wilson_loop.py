import matplotlib
from matplotlib import pyplot as plt
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
    N=4
    beta=5.5
    beta_imp=1.719
    u0=0.797
    imp=0 #default use of Wilson action
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                for t in range(0,N):
                    for mi in range(0,4):
                        gamma=Gamma(U,mi,x,y,z,t) #compute Gamma
                        if imp==1: #condition to use the improved action
                            gamma_imp=Gamma_improved(U,mi,x,y,z,t) #compute Gamma improved if it is asked
                        for p in range(10): #iteration before changing site of the lattice
                            s=random.randint(2,2*Nmatrix) #Choose a random matrix
                            dS = -beta/(3)*real(trace(rXc((rXc(M[s].copy(),U[x,y,z,t,mi].copy())-U[x,y,z,t,mi].copy()),gamma.copy()))) # change in action
                            if imp==1: #condition to use the improved action
                                dS = -beta_imp/(3)*(5/(3*u0**4)*real(trace(rXc((rXc(M[s],U[x,y,z,t,mi])-U[x,y,z,t,mi]),gamma)))-1/(12*u0**6)*real(trace(rXc((rXc(M[s],U[x,y,z,t,mi])-U[x,y,z,t,mi]),gamma_imp)))) # change in the improved action
                            if dS<0 or exp(-dS)>random.uniform(0,1):
                                U[x,y,z,t,mi] = rXc(M[s].copy(),U[x,y,z,t,mi].copy())  # update U

#Function that compute gamma for QCD using the Wilson action
#input:-x,y,z,t: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma(U,mi,x,y,z,t):
    N=4
    gamma=0. #inizializing gamma
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

#Function that compute gamma for QCD using the improved action using the rctangle terms
#input:-x,y,z,t: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma_improved(U,mi,x,y,z,t):
    N=4
    gamma=0. #inizializing gamma
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
                #next site on 2mi
                tp2mi=(t+2)%N
                xp2mi=x
                yp2mi=y
                zp2mi=z
                #previous site on mi
                tmmi=(t-1)%N
                xmmi=x
                ymmi=y
                zmmi=z
                #next site on 2mi and previous on ni
                tp2mimni=(t+2)%N
                xp2mimni=x
                yp2mimni=y
                zp2mimni=z
                #next site on mi and previous on 2ni
                tpmim2ni=(t+1)%N
                xpmim2ni=x
                ypmim2ni=y
                zpmim2ni=z
                #previous on ni and mi
                tmmimni=(t-1)%N
                xmmimni=x
                ymmimni=y
                zmmimni=z
                #next site on ni and previous on mi
                tmmipni=(t-1)%N
                xmmipni=x
                ymmipni=y
                zmmipni=z
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
                #next site on 2mi
                tp2mi=t
                xp2mi=(x+2)%N
                yp2mi=y
                zp2mi=z
                #previous site on mi
                tmmi=t
                xmmi=(x-1)%N
                ymmi=y
                zmmi=z
                #next site on 2mi and previous on ni
                tp2mimni=t
                xp2mimni=(x+2)%N
                yp2mimni=y
                zp2mimni=z
                #next site on mi and previous on 2ni
                tpmim2ni=t
                xpmim2ni=(x+1)%N
                ypmim2ni=y
                zpmim2ni=z
                #previous on ni and mi
                tmmimni=t
                xmmimni=(x-1)%N
                ymmimni=y
                zmmimni=z
                #next site on ni and previous on mi
                tmmipni=t
                xmmipni=(x-1)%N
                ymmipni=y
                zmmipni=z
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
                #next site on 2mi
                tp2mi=t
                xp2mi=x
                yp2mi=(y+2)%N
                zp2mi=z
                #previous site on mi
                tmmi=t
                xmmi=x
                ymmi=(y-1)%N
                zmmi=z
                #next site on 2mi and previous on ni
                tp2mimni=t
                xp2mimni=x
                yp2mimni=(y+2)%N
                zp2mimni=z
                #next site on mi and previous on 2ni
                tpmim2ni=t
                xpmim2ni=x
                ypmim2ni=(y+1)%N
                zpmim2ni=z
                #previous on ni and mi
                tmmimni=t
                xmmimni=x
                ymmimni=(y-1)%N
                zmmimni=z
                #next site on ni and previous on mi
                tmmipni=t
                xmmipni=x
                ymmipni=(y-1)%N
                zmmipni=z
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
                #next site on 2mi
                tp2mi=t
                xp2mi=x
                yp2mi=y
                zp2mi=(z+2)%N
                #previous site on mi
                tmmi=t
                xmmi=x
                ymmi=y
                zmmi=(z-1)%N
                #next site on 2mi and previous on ni
                tp2mimni=t
                xp2mimni=x
                yp2mimni=y
                zp2mimni=(z+2)%N
                #next site on mi and previous on 2ni
                tpmim2ni=t
                xpmim2ni=x
                ypmim2ni=y
                zpmim2ni=(z+1)%N
                #previous on ni and mi
                tmmimni=t
                xmmimni=x
                ymmimni=y
                zmmimni=(z-1)%N
                #next site on ni and previous on mi
                tmmipni=t
                xmmipni=x
                ymmipni=y
                zmmipni=(z-1)%N
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
                #next site on 2ni
                tp2ni=(t+2)%N
                xp2ni=x
                yp2ni=y
                zp2ni=z
                #Previous site on 2ni
                tm2ni=(t-2)%N
                xm2ni=x
                ym2ni=y
                zm2ni=z
                #next site on 2mi and previous on ni
                tp2mimni=(tp2mimni-1)%N
                xp2mimni=xp2mimni
                yp2mimni=yp2mimni
                zp2mimni=zp2mimni
                #next site on mi and previous on 2ni
                tpmim2ni=(tpmim2ni-2)%N
                xpmim2ni=xpmim2ni
                ypmim2ni=ypmim2ni
                zpmim2ni=zpmim2ni
                #previous on ni and mi
                tmmimni=(tmmimni-1)%N
                xmmimni=xmmimni
                ymmimni=ymmimni
                zmmimni=zmmimni
                #next site on ni and previous on mi
                tmmipni=(tmmipni+1)%N
                xmmipni=xmmipni
                ymmipni=ymmipni
                zmmipni=zmmipni
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
                #next site on 2ni
                tp2ni=t
                xp2ni=(x+2)%N
                yp2ni=y
                zp2ni=z
                #previous site on 2ni
                tm2ni=t
                xm2ni=(x-2)%N
                ym2ni=y
                zm2ni=z
                #next site on 2mi and previous on ni
                tp2mimni=tp2mimni
                xp2mimni=(xp2mimni-1)%N
                yp2mimni=yp2mimni
                zp2mimni=zp2mimni
                #next site on mi and previous on 2ni
                tpmim2ni=tpmim2ni
                xpmim2ni=(xpmim2ni-2)%N
                ypmim2ni=ypmim2ni
                zpmim2ni=zpmim2ni
                #previous on ni and mi
                tmmimni=tmmimni
                xmmimni=(xmmimni-1)%N
                ymmimni=ymmimni
                zmmimni=zmmimni
                #next site on ni and previous on mi
                tmmipni=tmmipni
                xmmipni=(xmmipni+1)%N
                ymmipni=ymmipni
                zmmipni=zmmipni
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
                #next site on 2ni
                tp2ni=t
                xp2ni=x
                yp2ni=(y+2)%N
                zp2ni=z
                #Previous site on 2ni
                tm2ni=t
                xm2ni=x
                ym2ni=(y-2)%N
                zm2ni=z
                #next site on 2mi and previous on ni
                tp2mimni=tp2mimni
                xp2mimni=xp2mimni
                yp2mimni=(yp2mimni-1)%N
                zp2mimni=zp2mimni
                #next site on mi and previous on 2ni
                tpmim2ni=tpmim2ni
                xpmim2ni=xpmim2ni
                ypmim2ni=(ypmim2ni-2)%N
                zpmim2ni=zpmim2ni
                #previous on ni and mi
                tmmimni=tmmimni
                xmmimni=(xmmimni-1)%N
                ymmimni=ymmimni
                zmmimni=zmmimni
                #next site on ni and previous on mi
                tmmipni=tmmipni
                xmmipni=xmmipni
                ymmipni=(ymmipni+1)%N
                zmmipni=zmmipni
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
                #next site on 2ni
                tp2ni=t
                xp2ni=x
                yp2ni=y
                zp2ni=(z+2)%N
                #previous site on 2ni
                tm2ni=t
                xm2ni=x
                ym2ni=y
                zm2ni=(z-2)%N
                #next site on 2mi and previous on ni
                tp2mimni=tp2mimni
                xp2mimni=xp2mimni
                yp2mimni=yp2mimni
                zp2mimni=(zp2mimni-1)%N
                #next site on mi and previous on 2ni
                tpmim2ni=tpmim2ni
                xpmim2ni=xpmim2ni
                ypmim2ni=ypmim2ni
                zpmim2ni=(zpmim2ni-2)%N
                #previous on ni and mi
                tmmimni=tmmimni
                xmmimni=xmmimni
                ymmimni=ymmimni
                zmmimni=(zmmimni-1)%N
                #next site on ni and mi
                tmmipni=tmmipni
                xmmipni=xmmipni
                ymmipni=ymmipni
                zmmipni=(zmmipni+1)%N

            gamma=gamma+rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,mi],U[xp2mi,yp2mi,zp2mi,tp2mi,ni]),rXc(dagger(U[xpmipni,ypmipni,zpmipni,tpmipni,mi]),rXc(dagger(U[xpni,ypni,zpni,tpni,mi]),dagger(U[x,y,z,t,ni]))))
            gamma=gamma+rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,mi],dagger(U[xp2mimni,yp2mimni,zp2mimni,tp2mimni,ni])),rXc(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,mi]),rXc(dagger(U[xmni,ymni,zmni,tmni,mi]),U[xmni,ymni,zmni,tmni,ni])))
            gamma=gamma+rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,ni],U[xpmipni,ypmipni,zpmipni,tpmipni,ni]),rXc(dagger(U[xp2ni,yp2ni,zp2ni,tp2ni,mi]),rXc(dagger(U[xpni,ypni,zpni,tpni,ni]),dagger(U[x,y,z,t,ni]))))
            gamma=gamma+rXc(rXc(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,ni]),dagger(U[xpmim2ni,ypmim2ni,zpmim2ni,tpmim2ni,ni])),rXc(dagger(U[xm2ni,ym2ni,zm2ni,tm2ni,mi]),rXc(U[xm2ni,ym2ni,zm2ni,tm2ni,ni],U[xmni,ymni,zmni,tmni,ni])))
            gamma=gamma+rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,ni],dagger(U[xpni,ypni,zpni,tpni,mi])),rXc(dagger(U[xmmipni,ymmipni,zmmipni,tmmipni,mi]),rXc(dagger(U[xmmi,ymmi,zmmi,tmmi,ni]),U[xmmi,ymmi,zmmi,tmmi,mi])))
            gamma=gamma+rXc(rXc(dagger(U[xpmimni,ypmimni,zpmimni,tpmimni,ni]),dagger(U[xmni,ymni,zmni,tmni,mi])),rXc(dagger(U[xmmimni,ymmimni,zmmimni,tmmimni,mi]),rXc(U[xmmimni,ymmimni,zmmimni,tmmimni,ni],U[xmmi,ymmi,zmmi,tmmi,mi])))
    return gamma.copy()

#Function that compute the Wilson axa Loop for each point of the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -x,y,z,t:position computed
#inner parameters:-N:points in the lattice
def compute_WL(U,x,y,z,t):
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
        for ni in range(0,4):
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

#Function that compute the Wilson Loop ax2a for each point of the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -x,y,z,t:position computed
#inner parameters:-N:points in the lattice
def compute_WLax2a(U,x,y,z,t):
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
        for ni in range(0,4):
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
                    #next site on 2ni
                    tp2ni=(t+2)%N
                    xp2ni=x
                    yp2ni=y
                    zp2ni=z
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
                    #next site on 2ni
                    tp2ni=t
                    xp2ni=(x+2)%N
                    yp2ni=y
                    zp2ni=z
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
                    #next site on 2ni
                    tp2ni=t
                    xp2ni=x
                    yp2ni=(y+2)%N
                    zp2ni=z
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
                    #next site on 2ni
                    tp2ni=t
                    xp2ni=x
                    yp2ni=y
                    zp2ni=(z+2)%N
                WL=WL+trace(rXc(U[x,y,z,t,mi].copy(),\
                        rXc(rXc(U[xpmi,ypmi,zpmi,tpmi,ni].copy(),U[xpmipni,ypmipni,zpmipni,tpmipni,ni].copy()),   \
                        rXc(dagger(U[xp2ni,yp2ni,zp2ni,tp2ni,mi].copy()), \
                        rXc(dagger(U[xpni,ypni,zpni,tpni,ni].copy()),dagger(U[x,y,z,t,ni].copy()))))))

    return real(WL)/(3.*12)




#allocation of the arrays and definition of the parameters
a=0.25
N=4
Nmatrix=200
N_cf=10
N_cor=20
U=zeros((N,N,N,N,4,3,3),dtype=complex) # inizializing U
WL=ones((N_cf), 'double')
WLax2=ones((N_cf), 'double')
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

for j in range(0,5*N_cor): # thermalize U
    update(U,M)
print('Termalized')
#Computation of the Wilson Loop for every point in the lattice and for every
#configuration
print('Iteration','Wilson loop for axa','Wilson loop for ax2a')
for alpha in range(0,N_cf): # loop on random paths
    for j in range(0,N_cor):
        update(U,M)
    WL[alpha]=0.
    WLax2[alpha]=0.
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                for t in range(0,N):
                    WL[alpha] =WL[alpha]+compute_WL(U,x,y,z,t) #axa Wilson loop
                    WLax2[alpha]=WLax2[alpha]+compute_WLax2a(U,x,y,z,t) #ax2a Wilson loop
    WL[alpha]=WL[alpha]/N**4
    WLax2[alpha]=WLax2[alpha]/N**4
    print(alpha+1,WL[alpha],WLax2[alpha]) #print of the results for each configuration

avg_WL=0.
avg_WLax2=0.
avg_WLSQ=0.
avg_WLax2SQ=0.
for alpha in range(0,N_cf): # compute MC averages
    avg_WL= avg_WL+WL[alpha]
    avg_WLax2=avg_WLax2+WLax2[alpha]
    avg_WLSQ=avg_WLSQ+WL[alpha]**2
    avg_WLax2SQ=avg_WLax2SQ+WLax2[alpha]**2
avg_WL=avg_WL/N_cf  #mean value on every configuration
avg_WLax2=avg_WLax2/N_cf
avg_WLSQ=avg_WLSQ/N_cf
avg_WLax2SQ=avg_WLax2SQ/N_cf
err_avg_WL=(abs(avg_WLSQ-avg_WL**2)/N_cf)**(1/2) #statistical error
err_avg_WLax2=(abs(avg_WLax2SQ-avg_WLax2**2)/N_cf)**(1/2) #statistical error

#print of the results
print('Mean value of the Wilson loop aX2a:',avg_WL,'+-',err_avg_WL)
print('Mean value of the Wilson loop aX2a:',avg_WLax2,'+-',err_avg_WLax2)
