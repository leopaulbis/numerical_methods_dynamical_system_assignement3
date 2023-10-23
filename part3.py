>import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

dim=2

#function of the system
#this function is function of R^n to R^n where n=dim
def f(x):
    return(np.array([x[1],-np.sin(x[0])]))

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

def first_int(x):
    return(x[1]**2/2-np.cos(x[0]))

x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# Compute the value of first_int
Z = first_int([X, Y])

# Create the plot of the levels curves
plt.contour(X, Y, Z, levels=20)  #levels =number of levels cruves

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Level curves of E(x,y)')


plt.colorbar()
plt.show()

###computation of the unstable and stable manifolds
u_p=np.array([1,1]) #eigen vector for 1
u_m=np.array([1,-1]) #eigen vector for -1
ini1=np.array([-np.pi,0])
ini2=np.array([np.pi,0])
##computation of the heteroclinic orbits
#first heteroclinic orbit

x0=ini1+1e-6*u_p
sol=solve_system(0.01,(dim,f),'RK45',30,0,x0)
sol_x=sol.y[0]
sol_y=sol.y[1]

print(sol_x)
print(sol_y)
plt.plot(sol_x,sol_y,'black',label="heteroclinic orbit from (-pi,0) to (pi,0)")



#second heteroclinic orbit

x0=ini2-1e-6*u_p
sol=solve_system(0.01,(dim,f),'RK45',30,0,x0)
sol_x=sol.y[0]
sol_y=sol.y[1]
plt.plot(sol_x,sol_y,'b',label="heteroclinic orbit from (pi,0) to (-pi,0)")
plt.scatter(np.pi,0,s=5,c='r')
plt.scatter(-np.pi,0,s=5,c='r')
plt.legend()
plt.show()

