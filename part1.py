import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

dim=2

#function of the system
#this function is function of R^n to R^n where n=dim
def f(x):
    return(np.array([x[1],-x[0]]))

#define the system of ode
def system_ode(t,x,dim,g):
    y=np.zeros(dim) #contain all the values of x_1,x_2...etc
    derivative=np.zeros(dim)

    for i in range (dim): #set all the initial conditions
        y[i]=x[i]

    for i in range(dim):
        derivative[i]=f(x)[i]

    return(derivative)

#return the solution with a max_step beetween each time
def solve_system(max_step,args,method,t_max,t_0,x_0):

    #t_eval = np.linspace(t_0, t_max, num_points) si on veut un nombre de point donné, rentrer t_eval à la place de max_step

    solution=solve_ivp(system_ode,[t_0,t_max],x_0,method=method,args=args,max_step=max_step)

    #print(f"x(0)={solution.y[0,0]},y(0)={solution.y[1,0]}")
    #print(f"x(2pi)={solution.y[0,-1]}y(2pi)={solution.y[1,-1]}")

    return(solution)

#return the solution with np point
def solve_system_NP(NP,args,method,t_max,t_0,x_0):
    time=np.linspace(t_0,t_max,NP)

    solution=solve_ivp(system_ode,[t_0,t_max],x_0,method=method,args=args,t_eval=time,max_step=0.001)

    return(solution)


def g(x):
    return(x[1])

def grad_g(x):
    grad=np.zeros(2)
    grad[1]=1
    return(grad)


def delta(t,sol):
    prod=np.dot(grad_g(sol),f(sol))
    return(-g(sol)/prod)

def poincare_map(x0,t0,dir,eps=1e-13):
    plt.scatter(x0[0], x0[1], s=5)
    delta_t=delta(t0,x0)
    x=x0
    t=t0
    if dir==1:
        while delta_t<0:
            sol=solve_system(0.001,(dim,f),'RK45',t+0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1]])
            plt.scatter(x[0], x[1], s=5,c='red')
            t=t+0.1
            delta_t=delta(t,x)

        while delta_t>1:
            sol=solve_system(0.001,(dim,f),'RK45',t+0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1]])
            plt.scatter(x[0], x[1], s=5,c='blue')
            delta_t=delta(t,x)
            t=t+0.1
    else:
        while delta_t>0:
            sol=solve_system(0.001,(dim,f),'RK45',t-0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1]])
            plt.scatter(x[0], x[1], s=5,c='red')
            t=t-0.1
            delta_t=delta(t,x)

        while delta_t<-1:
            sol=solve_system(0.001,(dim,f),'RK45',t-0.1,t,x)
            x=np.array([sol.y[0,-1],sol.y[1,-1]])
            plt.scatter(x[0], x[1], s=5,c='blue')
            delta_t=delta(t,x)
            t=t-0.1

    while np.abs(g(x))>eps:

        delta_t=delta(t,x)
        #take in acount if it is forward or backward
        sol=solve_system(0.001,(dim,f),'RK45',t+delta_t,t,x)
        #selection of the solution
        x=np.array([sol.y[0,-1],sol.y[1,-1]])
        plt.scatter(x[0], x[1], s=5,c='green')
        t=t+delta_t
    plt.show()
    return(x,t)

#dir equals to 1 or -1 !!!!
def poincare_map_NP(x0,t0,NP,dir):
    l=[]#contains the np points on the Poincaré section
    time=[]
    x=x0
    t=t0
    while len(l)!=NP:
        point,t=poincare_map(x,t,dir)
        l.append(point)
        time.append(t)
        if dir==1:
            sol=solve_system(0.001,(dim,f),'RK45',t+0.1,t,point)
            t=t+0.1
        else:
            sol=solve_system(0.001,(dim,f),'RK45',t-0.1,t,point)
            t=t-0.1
        x=np.array([sol.y[0,-1],sol.y[1,-1]])
    return(l,time)


x0=np.array([1,0]) #initial condition
t0=0 #initial time


#poincare=poincare_map(x0,t0,1)
#print(poincare)
# print(g(poincare))
#print(poincare_map_NP(x0,t0,3,-1))
print(np.abs(2*np.pi+poincare_map_NP(x0,t0,3,-1)[1][2]))

