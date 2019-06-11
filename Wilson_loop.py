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
    M=zeros((2*Nmatrix,3,3),dtype=complex)
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
        M[s,1,0]=conjugate((-M[s,1,1]*conjugate(M[s,0,1])-conjugate(M[s,0,2])*M[s,1,2])/M[s,0,0])
        normalizconst=0.
        for j in range(0,3):
            normalizconst=normalizconst+M[s,1,j]*conjugate(M[s,1,j])
        normalizconst=sqrt(normalizconst)
        for j in range(0,3):
            M[s,1,j]=M[s,1,j]/normalizconst
        M[s,2,0]=conjugate(M[s,0,1])*conjugate(M[s,1,2])-conjugate(M[s,0,2])*conjugate(M[s,1,1])
        M[s,2,1]=conjugate(M[s,1,0])*conjugate(M[s,0,2])-conjugate(M[s,0,0])*conjugate(M[s,1,2])
        M[s,2,2]=conjugate(M[s,0,0])*conjugate(M[s,1,1])-conjugate(M[s,1,0])*conjugate(M[s,0,1])
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
    N=len(M)
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
    x=[0,0,0,0]
    y=[0,0,0,0]
    beta=5.5
    for x[0] in range(0,N):
        for x[1] in range(0,N):
            for x[2] in range(0,N):
                for x[3] in range(0,N):
                    for mi in range(0,4):
                        for n in range(0,3):
                            for i in range(0,3):
                                old_U[i,n] = U[x[0],x[1],x[2],x[3],mi,i,n] # save original value

                        s=random.randint(0,2*Nmatrix) #Choose a random matrix
                        y=x
                        #print(linalg.det( U[x[0],x[1],x[2],x[3],mi]))
                        gamma=Gamma(U,mi,y[0],y[1],y[2],y[3]) #compute Gamma
                        U[x[0],x[1],x[2],x[3],mi] = rXc(M[s],U[x[0],x[1],x[2],x[3],mi]) # update U
                        dS = -beta/(3)*real(trace(rXc((U[x[0],x[1],x[2],x[3],mi]-old_U),gamma))) # change in action
                        if dS>0and exp(-dS)<random.uniform(0,1):
                            U[x[0],x[1],x[2],x[3],mi] = old_U # restore old value


#Function that compute gamma for QCD using the easiest action
#input:-i,j,k,l: position in which gamma is computed
#      -U:array of link variables
#inner parameter:-N:total number of points in the lattice
def Gamma(U,mi,i,j,k,l):
    N=4
    Gamma=zeros((3,3),dtype=complex)
    xpni=[0,0,0,0]
    xpmi=[0,0,0,0]
    xpmipni=[0,0,0,0]
    xmni=[0,0,0,0]
    xpmimni=[0,0,0,0]
    y=[0,0,0,0]
    y[0]=i
    y[1]=j
    y[2]=k
    y[3]=l
    for ni in range(0,4):
        y[0]=i
        y[1]=j
        y[2]=k
        y[3]=l
        xpni=y
        xpmi=y
        xpmipni=y
        xmni=y
        xpmimni=y

        if ni!=mi :
            xpni[ni]=(y[ni]+1)%N # next site on ni
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xpmi[mi]=(y[mi]+1)%N # next site on mi
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xpmipni[mi]=(y[mi]+1)%N # next site on ni and mi
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xpmipni[ni]=(y[ni]+1)%N
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xmni[ni]=(y[ni]-1)%N # previous site on ni
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xpmimni[mi]=(y[mi]+1)%N # next site on mi and previous on ni
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xpmimni[ni]=(y[ni]-1)%N
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            Gamma=Gamma+rXc(rXc(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3],ni],dagger(U[xpmipni[0],xpmipni[1],xpmipni[2],xpmipni[3],mi])),dagger(U[y[0],y[1],y[2],y[3],ni]))
            Gamma=Gamma+rXc(rXc(dagger(U[xpmimni[0],xpmimni[1],xpmimni[2],xpmimni[3],ni]),dagger(U[xmni[0],xmni[1],xmni[2],xmni[3],mi])),U[xmni[0],xmni[1],xmni[2],xmni[3],ni])
    return Gamma

#Function that compute the Wilson Loop for each point of the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -i,j,k,l:position computed
#inner parameters:-N:points in the lattice
def compute_WL(U,i,j,k,l):
    N=4
    WL = 0
    xpni=[0,0,0,0]
    xpmi=[0,0,0,0]
    xpmipni=[0,0,0,0]
    y=[0,0,0,0]
    y[0]=i
    y[1]=j
    y[2]=k
    y[3]=l
    for mi in range(0,4):
        for ni in range(0,4):
            y[0]=i
            y[1]=j
            y[2]=k
            y[3]=l
            xpni=y
            xpmi=y
            xpmipni=y
            if ni!=mi :
                xpni[ni]=(y[ni]+1)%N # next site on ni
                y[0]=i
                y[1]=j
                y[2]=k
                y[3]=l
                xpmi[mi]=(y[mi]+1)%N # next site on mi
                y[0]=i
                y[1]=j
                y[2]=k
                y[3]=l
                xpmipni[mi]=(y[mi]+1)%N # next site on ni and mi
                y[0]=i
                y[1]=j
                y[2]=k
                y[3]=l
                xpmipni[ni]=(y[ni]+1)%N
                y[0]=i
                y[1]=j
                y[2]=k
                y[3]=l
                WL=WL+trace(rXc(U[y[0],y[1],y[2],y[3],mi],rXc(rXc(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3],ni],dagger(U[xpmipni[0],xpmipni[1],xpmipni[2],xpmipni[3],mi])),dagger(U[y[0],y[1],y[2],y[3],ni]))))

    return real(WL)/(3.*12)

#allocation of the arrays and definition of the parameters
a=0.25
N=4
Nmatrix=100
N_cf=10
N_cor=20
U=zeros((N,N,N,N,4,3,3),dtype=complex) # inizializing U
WL=ones((N_cf), 'double')
M=zeros((2*Nmatrix,3,3),'double') #inizializing the random matrix
x=[0,0,0,0]
# inizializing U with the identity which belongs to the SU(3) group
for x[0] in range(0,N):
    for x[1] in range(0,N):
        for x[2] in range(0,N):
            for x[3] in range(0,N):
                for mi in range(0,4):
                    for n in range(0,3):
                        U[x[0],x[1],x[2],x[3],mi,n,n]=1.
# Generation of the random matrix
M=randommatrixSU3(M)
#Computation of the Monte Carlo mean value for the propagator in every time
#in the lattice
for j in range(0,5*N_cor): # thermalize U
    update(U,M)
for alpha in range(0,N_cf): # loop on random paths
    for j in range(0,N_cor):
        update(U,M)
    x=[0,0,0,0]
    WL[alpha]=0.
    for x[0] in range(0,N):
        for x[1] in range(0,N):
            for x[2] in range(0,N):
                for x[3] in range(0,N):
                    WL[alpha] =WL[alpha]+compute_WL(U,x[0],x[1],x[2],x[3])

    WL[alpha]=WL[alpha]/N**4
    print(WL[alpha])
avg_WL = 0.
for alpha in range(0,N_cf): # compute MC averages
    avg_WL= avg_WL+WL[alpha]
avg_WL = avg_WL/N_cf  #mean value on every configuration
print(avg_WL) #print of the result
