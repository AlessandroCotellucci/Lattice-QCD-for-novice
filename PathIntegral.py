import matplotlib
from matplotlib import pyplot as plt
import vegas
import math
import random
from numpy import *

#Function, coming from the functools packet,that given a function gives back the function
#computed in one of the variables of the function
def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

#Exact solution of the armonic oscillator propagator in one dimension
#input:-x:position
#output:-the exact propagator
#inner parameters:-E0: energy ground state
#                 -T:time interval
def exact(x):
    E0=0.5
    T=4
    return math.exp(-E0*T)*(math.exp(-(x**2)/2)*(math.pi**(-1/4)))**2

#Function that compute the Feynmann's factor for the path integral in 7 dimension of an
#armonic oscillator with periodic boundary condition
#input:-y:the variable on which we have to integrate
#      -x:the periodic boundary
#output:-the function
#inner parameters:-N: number of discretization of the time interval
#                 -a=T/N: discretized time interval
#                 -m:mass of the armonic oscillator
def f(x,y):
    T=4.
    m=1.
    N=8
    a=T/N
    Slat = ((m/(2*a))*(y[0]-x)**2+a*((x)**2)/(2))
    for d in range(N-1):
        if d==N-2:
            Slat += ((m/(2*a))*(x-y[d])**2+a*((y[d])**2)/(2))
        else:
            Slat += ((m/(2*a))*(y[d+1]-y[d])**2+a*((y[d])**2)/(2))
    return math.exp(-(Slat))*((m/(a*2*math.pi))**(N/2))

#Allocation of the array for the computation
exa=ones(100, 'double')
y=ones(100, 'double')
result=ones(10, 'double')
z=ones(10, 'double')
err=ones(10, 'double')

#Computatio for 10 values of x of the monte Carlo approximation
#print the value of x, the result and the fraction between the numerical result
#and the exact solution
for s in range(10):
    z[s]=s*2/9
    integ = vegas.Integrator(7*[[-5, 5]])
    integration = integ(partial(f,z[s]), nitn=10, neval=100000)
    result[s]=integration.mean
    err[s]=integration.sdev
    print(z[s],result[s],err[s])

#Computation for 100 values of x of the exact solution
for s in range(100):
    y[s]=s*2/100
    exa[s] = exact(y[s])

#Plot of the exact and the numerical slution
plt.plot(y,exa,'b',label='Exact')
plt.errorbar(z, result, yerr=err,fmt='o', color='black', label='Numerical');
#plt.plot(z,result,'ro',label='Numerical')
plt.legend(loc='upper right')
plt.axis([0,2.0,0.0,0.1])
plt.xlabel('x')
plt.ylabel(r'$\langle x|e^{-Ht}|x\rangle$')
plt.title('Harmonic oscillator')
plt.show()
