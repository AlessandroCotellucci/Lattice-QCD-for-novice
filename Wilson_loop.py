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
#input:-M:Matrix array
#output:-M:Matrix array with random SU(3) matrices
#innner parameters: -Nmatrix: number of matrices that we want generate
#                   -eps: small parameter for the Taylor serie
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
#      -M:Random matrix belonging to the group SU(3)
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
#      -ro:direction in which Gamma is computed
#inner parameter:-N:total number of points in the lattice
def Gamma(U,ro,x,y,z,t):
    N=8
    gamma=0. #inizializing gamma
    inc_ro=zeros((4),'int')
    inc_ni=zeros((4),'int')
    inc_ro[ro]=1 #increment in the ro direction
    #next site on mi
    xp_ro=(x+inc_ro[0].copy())%N
    yp_ro=(y+inc_ro[1].copy())%N
    zp_ro=(z+inc_ro[2].copy())%N
    tp_ro=(t+inc_ro[3].copy())%N
    for ni in range(0,4):
        if ni!=ro :
            inc_ni[ni]=1 #increment in the ni direction
            #next site on ni
            xp_ni=(x+inc_ni[0].copy())%N
            yp_ni=(y+inc_ni[1].copy())%N
            zp_ni=(z+inc_ni[2].copy())%N
            tp_ni=(t+inc_ni[3].copy())%N
            #Previous site on ni
            xm_ni=(x-inc_ni[0].copy())%N
            ym_ni=(y-inc_ni[1].copy())%N
            zm_ni=(z-inc_ni[2].copy())%N
            tm_ni=(t-inc_ni[3].copy())%N
            #next site on ni and mi
            xp_ro_p_ni=(x+inc_ro[0].copy()+inc_ni[0].copy())%N
            yp_ro_p_ni=(y+inc_ro[1].copy()+inc_ni[1].copy())%N
            zp_ro_p_ni=(z+inc_ro[2].copy()+inc_ni[2].copy())%N
            tp_ro_p_ni=(t+inc_ro[3].copy()+inc_ni[3].copy())%N
            #next site on mi and previous on ni
            xp_ro_m_ni=(x+inc_ro[0].copy()-inc_ni[0].copy())%N
            yp_ro_m_ni=(y+inc_ro[1].copy()-inc_ni[1].copy())%N
            zp_ro_m_ni=(z+inc_ro[2].copy()-inc_ni[2].copy())%N
            tp_ro_m_ni=(t+inc_ro[3].copy()-inc_ni[3].copy())%N
            inc_ni[ni]=0
            gamma=gamma+dot(dot(U[xp_ro,yp_ro,zp_ro,tp_ro,ni],dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,ro])),dagger(U[x,y,z,t,ni])) \
                        +dot(dot(dagger(U[xp_ro_m_ni,yp_ro_m_ni,zp_ro_m_ni,tp_ro_m_ni,ni]),dagger(U[xm_ni,ym_ni,zm_ni,tm_ni,ro])),U[xm_ni,ym_ni,zm_ni,tm_ni,ni])

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
    xp_mi=(x+incmi[0].copy())%N
    yp_mi=(y+incmi[1].copy())%N
    zp_mi=(z+incmi[2].copy())%N
    tp_mi=(t+incmi[3].copy())%N
    #previous site on mi
    xm_mi=(x-incmi[0].copy())%N
    ym_mi=(y-incmi[1].copy())%N
    zm_mi=(z-incmi[2].copy())%N
    tm_mi=(t-incmi[3].copy())%N
    #next site on 2mi
    xp_2mi=(x+2*incmi[0].copy())%N
    yp_2mi=(y+2*incmi[1].copy())%N
    zp_2mi=(z+2*incmi[2].copy())%N
    tp_2mi=(t+2*incmi[3].copy())%N

    for ni in range(0,4):
        if ni!=mi :
            incni[ni]=1
            #next site on ni
            xp_ni=(x+incni[0].copy())%N
            yp_ni=(y+incni[1].copy())%N
            zp_ni=(z+incni[2].copy())%N
            tp_ni=(t+incni[3].copy())%N
            #Previous site on ni
            xm_ni=(x-incni[0].copy())%N
            ym_ni=(y-incni[1].copy())%N
            zm_ni=(z-incni[2].copy())%N
            tm_ni=(t-incni[3].copy())%N
            #next site on ni and mi
            xp_mi_p_ni=(x+incmi[0].copy()+incni[0].copy())%N
            yp_mi_p_ni=(y+incmi[1].copy()+incni[1].copy())%N
            zp_mi_p_ni=(z+incmi[2].copy()+incni[2].copy())%N
            tp_mi_p_ni=(t+incmi[3].copy()+incni[3].copy())%N
            #next site on mi and previous on ni
            xp_mi_m_ni=(x+incmi[0].copy()-incni[0].copy())%N
            yp_mi_m_ni=(y+incmi[1].copy()-incni[1].copy())%N
            zp_mi_m_ni=(z+incmi[2].copy()-incni[2].copy())%N
            tp_mi_m_ni=(t+incmi[3].copy()-incni[3].copy())%N
            #next site on 2ni
            xp_2ni=(x+2*incni[0].copy())%N
            yp_2ni=(y+2*incni[1].copy())%N
            zp_2ni=(z+2*incni[2].copy())%N
            tp_2ni=(t+2*incni[3].copy())%N
            #Previous site on 2ni
            xm_2ni=(x-2*incni[0].copy())%N
            ym_2ni=(y-2*incni[1].copy())%N
            zm_2ni=(z-2*incni[2].copy())%N
            tm_2ni=(t-2*incni[3].copy())%N
            #next site on 2mi and previous on ni
            xp_2mi_m_ni=(x+2*incmi[0].copy()-incni[0].copy())%N
            yp_2mi_m_ni=(y+2*incmi[1].copy()-incni[1].copy())%N
            zp_2mi_m_ni=(z+2*incmi[2].copy()-incni[2].copy())%N
            tp_2mi_m_ni=(t+2*incmi[3].copy()-incni[3].copy())%N
            #next site on mi and previous on 2ni
            xp_mi_m_2ni=(x+incmi[0].copy()-2*incni[0].copy())%N
            yp_mi_m_2ni=(y+incmi[1].copy()-2*incni[1].copy())%N
            zp_mi_m_2ni=(z+incmi[2].copy()-2*incni[2].copy())%N
            tp_mi_m_2ni=(t+incmi[3].copy()-2*incni[3].copy())%N
            #previous site on ni and mi
            xm_mi_m_ni=(x-incmi[0].copy()-incni[0].copy())%N
            ym_mi_m_ni=(y-incmi[1].copy()-incni[1].copy())%N
            zm_mi_m_ni=(z-incmi[2].copy()-incni[2].copy())%N
            tm_mi_m_ni=(t-incmi[3].copy()-incni[3].copy())%N
            #next site on ni and previous mi
            xm_mi_p_ni=(x-incmi[0].copy()+incni[0].copy())%N
            ym_mi_p_ni=(y-incmi[1].copy()+incni[1].copy())%N
            zm_mi_p_ni=(z-incmi[2].copy()+incni[2].copy())%N
            tm_mi_p_ni=(t-incmi[3].copy()+incni[3].copy())%N
            incni[ni]=0

            gamma_imp=gamma_imp+dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,mi],U[xp_2mi,yp_2mi,zp_2mi,tp_2mi,ni]),dot(dagger(U[xp_mi_p_ni,yp_mi_p_ni,zp_mi_p_ni,tp_mi_p_ni,mi]),dot(dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,mi]),dagger(U[x,y,z,t,ni])))) \
                                +dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,mi],dagger(U[xp_2mi_m_ni,yp_2mi_m_ni,zp_2mi_m_ni,tp_2mi_m_ni,ni])),dot(dagger(U[xp_mi_m_ni,yp_mi_m_ni,zp_mi_m_ni,tp_mi_m_ni,mi]),dot(dagger(U[xm_ni,ym_ni,zm_ni,tm_ni,mi]),U[xm_ni,ym_ni,zm_ni,tm_ni,ni]))) \
                                +dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],U[xp_mi_p_ni,yp_mi_p_ni,zp_mi_p_ni,tp_mi_p_ni,ni]),dot(dagger(U[xp_2ni,yp_2ni,zp_2ni,tp_2ni,mi]),dot(dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,ni]),dagger(U[x,y,z,t,ni])))) \
                                +dot(dot(dagger(U[xp_mi_m_ni,yp_mi_m_ni,zp_mi_m_ni,tp_mi_m_ni,ni]),dagger(U[xp_mi_m_2ni,yp_mi_m_2ni,zp_mi_m_2ni,tp_mi_m_2ni,ni])),dot(dagger(U[xm_2ni,ym_2ni,zm_2ni,tm_2ni,mi]),dot(U[xm_2ni,ym_2ni,zm_2ni,tm_2ni,ni],U[xm_ni,ym_ni,zm_ni,tm_ni,ni]))) \
                                +dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,mi])),dot(dagger(U[xm_mi_p_ni,ym_mi_p_ni,zm_mi_p_ni,tm_mi_p_ni,mi]),dot(dagger(U[xm_mi,ym_mi,zm_mi,tm_mi,ni]),U[xm_mi,ym_mi,zm_mi,tm_mi,mi]))) \
                                +dot(dot(dagger(U[xp_mi_m_ni,yp_mi_m_ni,zp_mi_m_ni,tp_mi_m_ni,ni]),dagger(U[xm_ni,ym_ni,zm_ni,tm_ni,mi])),dot(dagger(U[xm_mi_m_ni,ym_mi_m_ni,zm_mi_m_ni,tm_mi_m_ni,mi]),dot(U[xm_mi_m_ni,ym_mi_m_ni,zm_mi_m_ni,tm_mi_m_ni,ni],U[xm_mi,ym_mi,zm_mi,tm_mi,mi])))
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
        xp_mi=(x+incmi[0].copy())%N
        yp_mi=(y+incmi[1].copy())%N
        zp_mi=(z+incmi[2].copy())%N
        tp_mi=(t+incmi[3].copy())%N
        for ni in range(0,mi):
            incni[ni]=1 #increment in the ni direction
            #next site on ni
            xp_ni=(x+incni[0].copy())%N
            yp_ni=(y+incni[1].copy())%N
            zp_ni=(z+incni[2].copy())%N
            tp_ni=(t+incni[3].copy())%N
            incni[ni]=0

            WL=WL+trace(dot(U[x,y,z,t,mi],dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,mi])),dagger(U[x,y,z,t,ni]))))
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
        xp_mi=(x+incmi[0].copy())%N
        yp_mi=(y+incmi[1].copy())%N
        zp_mi=(z+incmi[2].copy())%N
        tp_mi=(t+incmi[3].copy())%N
        for ni in range(3,mi,-1):
            if ni!=mi :
                incni[ni]=1 #increment in the ni direction
                #next site on ni
                xp_ni=(x+incni[0].copy())%N
                yp_ni=(y+incni[1].copy())%N
                zp_ni=(z+incni[2].copy())%N
                tp_ni=(t+incni[3].copy())%N
                #next site on ni and mi
                xp_mi_p_ni=(x+incmi[0].copy()+incni[0].copy())%N
                yp_mi_p_ni=(y+incmi[1].copy()+incni[1].copy())%N
                zp_mi_p_ni=(z+incmi[2].copy()+incni[2].copy())%N
                tp_mi_p_ni=(t+incmi[3].copy()+incni[3].copy())%N
                #next site on 2ni
                xp_2ni=(x+2*incni[0].copy())%N
                yp_2ni=(y+2*incni[1].copy())%N
                zp_2ni=(z+2*incni[2].copy())%N
                tp_2ni=(t+2*incni[3].copy())%N
                incni[ni]=0
                WL=WL+trace(dot(U[x,y,z,t,mi],\
                        dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],U[xp_mi_p_ni,yp_mi_p_ni,zp_mi_p_ni,tp_mi_p_ni,ni]),   \
                        dot(dagger(U[xp_2ni,yp_2ni,zp_2ni,tp_2ni,mi]), \
                        dot(dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,ni]),dagger(U[x,y,z,t,ni]))))))

        incmi[mi]=0
    return real(WL)/(3.*6.)



#main function
#allocation of the arrays and definition of the parameters
a=0.25 #spacing of the lattice
N=8 #number of points in the lattice
Nmatrix=200 #number of SU(3) random matrices
N_cf=10 #number of Montecarlo configurations
N_cor=50 #number of update before computing the Wilson loop's
U=zeros((N,N,N,N,4,3,3),dtype=complex) # inizializing U: link variables
WL=ones((N_cf), 'double') #inizializing WL: Wilson loop axa
WLax2=ones((N_cf), 'double') #inizializing WLax2: Wilson loop ax2a
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
