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

#Function that compute the Wilson axa Loop for each point of the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -x,y,z,t:position computed
#inner parameters:-N:points in the lattice
def compute_WL(U,x,y,z,t):
    N=8
    WL = 0.
    incmi=zeros((4),'int')
    incni=zeros((4),'int')
    for mi in range(0,4):
        incmi[mi]=1 #increment in the mi direction
        #next site on mi
        xpmi=(x+incmi[0].copy())%N
        ypmi=(y+incmi[1].copy())%N
        zpmi=(z+incmi[2].copy())%N
        tpmi=(t+incmi[3].copy())%N
        for ni in range(0,mi):
            incni[ni]=1 #increment in the ni direction
            #next site on ni
            xpni=(x+incni[0].copy())%N
            ypni=(y+incni[1].copy())%N
            zpni=(z+incni[2].copy())%N
            tpni=(t+incni[3].copy())%N
            incni[ni]=0

            WL=WL+trace(dot(U[x,y,z,t,mi],dot(dot(U[xpmi,ypmi,zpmi,tpmi,ni],dagger(U[xpni,ypni,zpni,tpni,mi])),dagger(U[x,y,z,t,ni]))))
        incmi[mi]=0
    return real(WL)/(3.*6.)

#Function that compute the Wilson Loop ax2a for each point of the lattice using the link variables
#generated using the Metropolis algoritm
#input:-U: array of the link variables
#      -x,y,z,t:position computed
#inner parameters:-N:points in the lattice
def compute_WLax2a(U,x,y,z,t):
    N=8
    WL = 0.
    incmi=zeros((4),'int')
    incni=zeros((4),'int')
    for mi in range(0,4):
        incmi[mi]=1 #increment in the mi direction
        #next site on mi
        xpmi=(x+incmi[0].copy())%N
        ypmi=(y+incmi[1].copy())%N
        zpmi=(z+incmi[2].copy())%N
        tpmi=(t+incmi[3].copy())%N
        for ni in range(3,mi,-1):
            if ni!=mi :
                incni[ni]=1 #increment in the ni direction
                #next site on ni
                xpni=(x+incni[0].copy())%N
                ypni=(y+incni[1].copy())%N
                zpni=(z+incni[2].copy())%N
                tpni=(t+incni[3].copy())%N
                #next site on ni and mi
                xpmipni=(x+incmi[0].copy()+incni[0].copy())%N
                ypmipni=(y+incmi[1].copy()+incni[1].copy())%N
                zpmipni=(z+incmi[2].copy()+incni[2].copy())%N
                tpmipni=(t+incmi[3].copy()+incni[3].copy())%N
                #next site on 2ni
                xp2ni=(x+2*incni[0].copy())%N
                yp2ni=(y+2*incni[1].copy())%N
                zp2ni=(z+2*incni[2].copy())%N
                tp2ni=(t+2*incni[3].copy())%N
                incni[ni]=0
                WL=WL+trace(dot(U[x,y,z,t,mi],\
                        dot(dot(U[xpmi,ypmi,zpmi,tpmi,ni],U[xpmipni,ypmipni,zpmipni,tpmipni,ni]),   \
                        dot(dagger(U[xp2ni,yp2ni,zp2ni,tp2ni,mi]), \
                        dot(dagger(U[xpni,ypni,zpni,tpni,ni]),dagger(U[x,y,z,t,ni]))))))

        incmi[mi]=0
    return real(WL)/(3.*6.)




#allocation of the arrays and definition of the parameters
a=0.25
N=8
Nmatrix=200
N_cf=10
N_cor=50
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

for j in range(0,2*N_cor): # thermalize U
    update(U,M)
print('Termalized')
#Computation of the Wilson Loop for every point in the lattice and for every
#configuration
file1 = open("Results.txt","w")
print('Iteration','Wilson loop for axa','Wilson loop for ax2a')
file1.write('Iteration    Wilson loop for axa    Wilson loop for ax2a')
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
    file1.write('\n')  #print on a file
    file1.write(str(alpha+1))
    file1.write('   ')
    file1.write(str(WL[alpha]))
    file1.write('   ')
    file1.write(str(WLax2[alpha]))
    file1.write('   ')
    print(alpha+1,WL[alpha],WLax2[alpha]) #print of the results for each configuration
file1.write('\n')
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
print('Mean value of the Wilson loop aXa:',avg_WL,'+-',err_avg_WL)
print('Mean value of the Wilson loop aX2a:',avg_WLax2,'+-',err_avg_WLax2)
file1.write('Mean value of the Wilson loop aX2a:  ')
file1.write(str(avg_WL))
file1.write('+-')
file1.write(str(err_avg_WL))
file1.write('\n')
file1.write('Mean value of the Wilson loop aX2a:  ')
file1.write(str(avg_WLax2))
file1.write('+-')
file1.write(str(err_avg_WLax2))
file1.close()
