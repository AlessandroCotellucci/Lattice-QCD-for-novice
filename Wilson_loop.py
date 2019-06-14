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
    eps=0.24
    for s in range(0,Nmatrix):
        for j in range(0,3):
            for i in range(0,3):
                M[s,j,i]=complex(random.uniform(-1,1),random.uniform(-1,1))
        for j in range(0,3):
            for i in range(0,3):
                M[j,i]=conjugate(M[i,j])
        for j in range(0,3):
            for i in range(0,3):
                M[s,j,i]=identity[j,i]+cmath.sqrt(-1)*eps*M[s,j,i]
        normalizconst=0.
        #Unitarizing
        for j in range(0,3):
            normalizconst=normalizconst+M[s,0,j]*conjugate(M[s,0,j])
        normalizconst=sqrt(normalizconst)
        for j in range(0,3):
            M[s,0,j]=M[s,0,j]/normalizconst

        M[s,1,0]=(-conjugate(M[s,1,1])*M[s,0,1]-M[s,0,2]*conjugate(M[s,1,2]))/conjugate(M[s,0,0])
        normalizconst=0.
        for j in range(0,3):
            normalizconst=normalizconst+M[s,1,j]*conjugate(M[s,1,j])
        normalizconst=sqrt(normalizconst)
        for j in range(0,3):
            M[s,1,j]=M[s,1,j]/normalizconst
        M[s,2,0]=conjugate(M[s,0,1])*conjugate(M[s,1,2])-conjugate(M[s,0,2])*conjugate(M[s,1,1])
        M[s,2,1]=conjugate(M[s,1,0])*conjugate(M[s,0,2])-conjugate(M[s,0,0])*conjugate(M[s,1,2])
        M[s,2,2]=conjugate(M[s,0,0])*conjugate(M[s,1,1])-conjugate(M[s,1,0])*conjugate(M[s,0,1])
        #M[s,2,0]=1./(M[s,0,1]*M[s,1,2]-M[s,1,1]*M[s,0,2])*(1-M[s,0,0]*M[s,1,1]*M[s,2,2]-M[s,0,2]*M[s,1,0]*M[s,2,1]+M[s,2,1]*M[s,1,2]*M[s,0,0]+M[s,2,2]*M[s,1,0]*M[s,0,1])
        #normalizconst=0.
        #Unitarizing
        #for j in range(0,3):
        #    normalizconst=normalizconst+M[s,2,j]*conjugate(M[s,2,j])
        #normalizconst=sqrt(normalizconst)
        #for j in range(0,3):
        #    M[s,2,j]=M[s,2,j]/normalizconst
        #print(conjugate(M[s,1,0])*M[s,1,0]+M[s,1,1]*conjugate(M[s,1,1])+M[s,1,2]*conjugate(M[s,1,2]))
        #print(conjugate(M[s,0,0])*M[s,0,0]+conjugate(M[s,0,1])*M[s,0,1]+conjugate(M[s,0,2])*M[s,0,2])
        #print(conjugate(M[s,1,0])*M[s,2,0]+M[s,2,1]*conjugate(M[s,1,1])+M[s,2,2]*conjugate(M[s,1,2]))
        #print(conjugate(M[s,0,0])*M[s,2,0]+conjugate(M[s,0,1])*M[s,2,1]+conjugate(M[s,0,2])*M[s,2,2])
        #print(conjugate(M[s,1,0])*M[s,0,0]+M[s,0,1]*conjugate(M[s,1,1])+M[s,0,2]*conjugate(M[s,1,2]))
        #print(conjugate(M[s,2,0])*M[s,2,0]+conjugate(M[s,2,1])*M[s,2,1]+conjugate(M[s,2,2])*M[s,2,2])
        #print('determinante',linalg.det(M[s]))
        M[s+Nmatrix]=dagger(M[s])
    return M

#Function that compute the dagger of a matrix
def dagger(M):
    N=len(M)
    H=zeros((N,N),dtype=complex)
    for i in range(0,N):
        for j in range(0,N):
            H[i,j]=conjugate(M[j,i])
    return H

#Function that compute the product of two square matrix with equal dimension
def rXc(M,H):
    N=3
    R=zeros((N,N),dtype=complex)
    for j in range(0,N):
        for i in range(0,N):
            for n in range(0,N):
                R[i,j]=R[i,j]+M[i,n]*H[n,j]
    return R


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
#in the action using the easiest action for QCD
#input:-U:array of the link variable
#inner parameters: -N:total number of points in the lattice
def update(U,M):
    Nmatrix=100
    N=4
    gamma=zeros((3,3),dtype=complex)
    old_U=zeros((3,3),dtype=complex)
    beta=5.5
    for t in range(0,N):
        for x in range(0,N):
            for y in range(0,N):
                for z in range(0,N):
                    for mi in range(0,4):
                        for n in range(0,3):
                            for i in range(0,3):
                                old_U[i,n] = U[t,x,y,z,mi,i,n] # save original value
                        s=random.randint(4,2*Nmatrix-2) #Choose a random matrix
                        gamma=Gamma(U,mi,t,x,y,z) #compute Gamma
                        U[t,x,y,z,mi] = rXc(M[s],U[t,x,y,z,mi]) # update U
                        deter=linalg.det(U[t,x,y,z,mi])
                        err=abs(1.-deter)
                        if err>1:
                            print(deter,linalg.det(M[s]),linalg.det(old_U),s)
                            print(U[t,x,y,z,mi],M[s],old_U)
                        dS = -beta/(3)*real(trace(rXc((U[t,x,y,z,mi]-old_U),gamma))) # change in action
                        if dS>0 and exp(-dS)<random.uniform(0,1):
                            U[t,x,y,z,mi] = old_U # restore old value


#Function that compute gamma for QCD using the easiest action
#input:-i,j,k,l: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma(U,mi,t,x,y,z):
    N=4
    Gamma=zeros((3,3),dtype=complex)

    for ni in range(0,4):
        #upload of the poin on mi
        if mi==0:
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
        if mi==1:
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
        if mi==2:
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
        if mi==3:
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
            if ni==0:
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
            if ni==1:
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
            if ni==2:
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
            if ni==3:
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

            Gamma=Gamma+rXc(rXc(U[tpmi,xpmi,ypmi,zpmi,ni],dagger(U[tpmipni,xpmipni,ypmipni,zpmipni,mi])),dagger(U[t,x,y,z,ni]))
            Gamma=Gamma+rXc(rXc(dagger(U[tpmimni,xpmimni,ypmimni,zpmimni,ni]),dagger(U[tmni,xmni,ymni,zmni,mi])),U[tmni,xmni,ymni,zmni,ni])
            
    return Gamma

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
        if mi==0:
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
        if mi==1:
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
        if mi==2:
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
        if mi==3:
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
                if ni==0:
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
                if ni==1:
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
                if ni==2:
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
                if ni==3:
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
                WL=WL+trace(rXc(U[t,x,y,z,mi],rXc(rXc(U[tpmi,xpmi,ypmi,zpmi,ni],dagger(U[tpmipni,xpmipni,ypmipni,zpmipni,mi])),dagger(U[t,x,y,z,ni]))))

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
for t in range(0,N):
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                for mi in range(0,4):
                    for n in range(0,3):
                        U[t,x,y,z,mi,n,n]=1.
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
    for t in range(0,N):
        for x in range(0,N):
            for y in range(0,N):
                for z in range(0,N):
                    WL[alpha] =WL[alpha]+compute_WL(U,t,x,y,z)

    WL[alpha]=WL[alpha]/N**4
    print(WL[alpha])
avg_WL = 0.
for alpha in range(0,N_cf): # compute MC averages
    avg_WL= avg_WL+WL[alpha]
avg_WL = avg_WL/N_cf  #mean value on every configuration
print(avg_WL) #print of the result
