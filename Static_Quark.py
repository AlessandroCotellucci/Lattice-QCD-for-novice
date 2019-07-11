import matplotlib
from matplotlib import pyplot as plt
import math
import cmath
import random
from numpy import *
from scipy.optimize import least_squares

#Function that compute the dagger of a matrix
def dagger(M):
    N=len(M)
    H=zeros((N,N),dtype=complex)
    R=matrix(M)
    H=R.getH()
    return H.copy()

#Function to compute the non liear regression
#input:-x:parameters of the function
#      -r:radial variables
#      -y:value on which we want to compute the fit
def exact(x,r,y):
    return x[0]*r-x[1]/r+x[2]-y

#Function to generate data with the static potential
#input:-r:radius
#      -a:linear parameter "String tension"
#      -b:Coulomb-like paraeter
#      -c:constant factor
def gen_data(r,a,b,c):
    return a*r-b/r+c


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
                H[j,i]=complex(random.uniform(-1,1),random.uniform(-1,1)) #random generation of the matriz
        H=(H.copy()+dagger(H.copy()))/2. #generation of the hermitian matrice
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
                            s=random.randint(2,2*Nmatrix-2) #Choose a random matrix
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
#      -mi:direction in which Gamma_improved is computed
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
    inc[f]=1 #increment in the f direction
    for i in range(n):
        x_inc=(x+i*inc[0])%N #increment in the Wilson line direction
        y_inc=(y+i*inc[1])%N #increment in the Wilson line direction
        z_inc=(z+i*inc[2])%N #increment in the Wilson line direction
        t_inc=(t+i*inc[3])%N #increment in the Wilson line direction
        productU=dot(productU,U[x_inc,y_inc,z_inc,t_inc,f]) #product of the successives link variables along the direction f
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
    inc[f]=1 #increment in the f direction
    for i in range(n):
        x_inc=(x-(i+1)*inc[0])%N #decrement in the Wilson line direction
        y_inc=(y-(i+1)*inc[1])%N #decrement in the Wilson line direction
        z_inc=(z-(i+1)*inc[2])%N #decrement in the Wilson line direction
        t_inc=(t-(i+1)*inc[3])%N #decrement in the Wilson line direction
        productUdagger=dot(productUdagger,dagger(U[x_inc,y_inc,z_inc,t_inc,f])) #product of the inverse link variables along the direction f
    return productUdagger.copy()

#Function that compute the Wilson Loop time*a x radius*a mean value on the lattice using the link variables
#generated using the Metropolis algoritm. Compute only the planar Wilson loops
#input:-U: array of the link variables
#      -time: temporal dimension of the loop
#      -radius: spatial dimension of the loop
#output:-WL: mean value of the Wilson loop on all the spatial direction and all the points of the lattice
#inner parameters:-N:points in the lattice
def compute_WL(U,time,radius):
    N=8
    #U=zeros((N,N,N,N,4,3,3),dtype=complex)
    productUt=zeros((3,3),dtype=complex) #allocation
    productUr=zeros((3,3),dtype=complex) #allocation
    productUtdagger=zeros((3,3),dtype=complex) #allocation
    productUrdagger=zeros((3,3),dtype=complex) #allocation
    inc=zeros((3),'int') #allocation
    WL=0.
    for space in range(3):
        #U=SubstitutionU(F,space) #Substitution with the smeared quantity
        inc[space]=1 #increment in the spatial direction
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for t in range(N):
                        #incremented coordinates in the spatial direction
                        x_inc=(x+radius*inc[0])%N
                        y_inc=(y+radius*inc[1])%N
                        z_inc=(z+radius*inc[2])%N
                        productUt=ProductU(U,x,y,z,t,time,3) #Wilson line in time direction
                        productUr=ProductU(U,x,y,z,(t+time)%N,radius,space) #Wilson line in space direction
                        productUtdagger=ProductUdagger(U,x_inc,y_inc,z_inc,(t+time)%N,time,3) #inverse Wilson line in time direction
                        productUrdagger=ProductUdagger(U,x_inc,y_inc,z_inc,t,radius,space) #inverse Wilson line in space direction
                        WL=WL+real(trace(dot(productUt,dot(productUr,dot(productUtdagger,productUrdagger))))) #compute of the Wilson loop
        inc[space]=0
    return abs(WL)/9./N**4



#Main program
#allocation of the arrays and definition of the parameters
a=0.25 #spacing of the lattice
N=8 #number of site in the lattice
N_sim=5
N_sim_t=8
Nmatrix=200 #total number of SU(3) matrices
N_cf=10 #number of montecarlo configuration
N_cor=50 #number of update before compute the result
U=zeros((N,N,N,N,4,3,3),dtype=complex) #inizializing U: link variables
WL=zeros((N_cf,N_sim_t,N_sim), 'double') #inizializing WL: Wilson loop rsults for every configuration
M=zeros((Nmatrix,3,3),dtype=complex) #inizializing the random matrix
avg_WL=zeros((N_sim_t,N_sim),'double') #inizializing avg_WL: mean value of the Wilson loop on the different configuration
avg_WLSQ=zeros((N_sim_t,N_sim),'double') #inizializing avg_WLSQ: mean value of the Wilson loop squared on the different configuration
err_avg_WL=zeros((N_sim_t,N_sim),'double') #inizializing err_avg_WL: statistical error of avg_WL
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

    for time in range(1,N_sim_t):
        for radius in range(1,N_sim):
            WL[alpha,time,radius]=compute_WL(U,time,radius) #compute WL

    print("End simulation number:",alpha+1)

for time in range(1,N_sim_t): #computation of the Wilson loop mean value for every possible values of time and radius
    for radius in range(1,N_sim):
        avg_WL[time,radius]=0.
        avg_WLSQ[time,radius]=0.
        for alpha in range(0,N_cf): # compute MC averages
            avg_WL[time,radius]= avg_WL[time,radius]+WL[alpha,time,radius]
            avg_WLSQ[time,radius]=avg_WLSQ[time,radius]+WL[alpha,time,radius]**2
        avg_WL[time,radius] = avg_WL[time,radius]/N_cf  #mean value on every configuration
        avg_WLSQ[time,radius] = avg_WLSQ[time,radius]/N_cf
        err_avg_WL[time,radius]=((avg_WLSQ[time,radius]-avg_WL[time,radius]**2)/N_cf)**(1/2) #statistical error

for i in range(0,N_sim-1): #computation of the fraction in the high time limit with the error
    radius=i+1
    potential[i]=avg_WL[N_sim_t-2,radius]/avg_WL[N_sim_t-1,radius] #computation of the potential
    err_potential[i]=((err_avg_WL[N_sim-1,radius]/avg_WL[N_sim-1,radius])**2+(err_avg_WL[N_sim-2,radius]/avg_WL[N_sim-2,radius])**2)**(1/2)*potential[i] #propagation of the error on the potential
    rad[i]=radius

#non linear regression using the fit function
x=ones(3)
res_lsq = least_squares(exact, x, args=(rad, potential))
rad_test = linspace(1, 5,100)
V_lsq = gen_data(rad_test, *res_lsq.x)

#plot of the numerical solution and the fit curve
plt.errorbar(rad, potential, yerr=err_potential, fmt='.', color='blue', label='Numerical');
plt.plot(rad_test,V_lsq,'b',label='Regression')
plt.legend(loc='upper right')
plt.title('Static quark potential')
plt.xlabel('$r/a$')
plt.ylabel('$V(r)a$')
plt.show()
