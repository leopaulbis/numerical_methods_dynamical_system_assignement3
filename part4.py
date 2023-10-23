import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

dim=2

#function of the system
#this function is function of R^n to R^n where n=dim
def f(x):
    return(np.array([3*x[0]-x[0]**2-2*x[1]*x[0],2*x[1]-x[1]*x[0]-x[1]**2]))

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

# ###Plot of the orbits around the equilibrium point
# ##(0,0)
#
#
#
# Initial conditions (change as needed)
x0 = np.linspace(0,0.1,3)
y0=np.linspace(0,0.1,3)

# Solve the system with a max_step and plot the orbits
max_step = 0.001

for i in range(len(x0)):
    for j in range(len(y0)):
        x_ini=np.array([x0[i],y0[j]])
        plt.scatter(x_ini[0],x_ini[1],s=1,c='b')
        print(f'x_ini={x_ini}')
        sol = solve_system(max_step, (dim,f), 'RK45', 1.5, 0, x_ini)
        plt.plot(sol.y[0],sol.y[1],label=f"x0={x_ini[0],x_ini[1]}")
plt.scatter(0,0,s=5,c='r')
plt.legend()
plt.show()

# ##(3,0)
#
# # # Initial conditions (change as needed)
# x0 = np.linspace(2.9,3.1,3)
# y0=np.linspace(0,0.1,3)
#
# # Solve the system with a max_step and plot the orbits
# max_step = 0.001
#
# for i in range(len(x0)):
#     for j in range(len(y0)):
#         x_ini=np.array([x0[i],y0[j]])
#         plt.scatter(x_ini[0],x_ini[1],s=1,c='b')
#         print(f'x_ini={x_ini}')
#         sol = solve_system(max_step, (dim,f), 'RK45', 5, 0, x_ini)
#         plt.plot(sol.y[0],sol.y[1],label=f"x0={x_ini[0],x_ini[1]}")
# plt.scatter(3,0,s=10,c='r')
# plt.legend()
# plt.show()
#
# ##(0,2)
#
# # Initial conditions (change as needed)
# y0 = np.linspace(1.9,2.1,3)
# x0=np.linspace(0,0.1,3)
#
# # Solve the system with a max_step and plot the orbits
# max_step = 0.001
#
# for i in range(len(x0)):
#     for j in range(len(y0)):
#         x_ini=np.array([x0[i],y0[j]])
#         plt.scatter(x_ini[0],x_ini[1],s=1,c='b')
#         print(f'x_ini={x_ini}')
#         sol = solve_system(max_step, (dim,f), 'RK45', 5, 0, x_ini)
#         plt.plot(sol.y[0],sol.y[1],label=f"x0={x_ini[0],x_ini[1]}")
# plt.scatter(0,2,s=10,c='r')
# plt.legend()
# plt.show()
#
# ##(1,1)
#
# # Initial conditions (change as needed)
# y0 = np.linspace(0.6,1.9,4)
# x0=np.linspace(0.6,1.9,4)
#
# # Solve the system with a max_step and plot the orbits
# max_step = 0.001
#
# for i in range(len(x0)):
#     for j in range(len(y0)):
#         x_ini=np.array([x0[i],y0[j]])
#         plt.scatter(x_ini[0],x_ini[1],s=1,c='b')
#         print(f'x_ini={x_ini}')
#         sol = solve_system(max_step, (dim,f), 'RK45', 7, 0, x_ini)
#         plt.plot(sol.y[0],sol.y[1],label=f"x0={x_ini[0],x_ini[1]}")
# plt.scatter(1,1,s=10,c='r')
# #plt.legend()
# plt.show()

##global dynamic

# Initial conditions (change as needed)
y0 = np.linspace(0,4,8)
x0=np.linspace(0,4,8)

#Solve the system with a max_step and plot the orbits
max_step = 0.001

# for i in range(len(x0)):
#     for j in range(len(y0)):
#         x_ini=np.array([x0[i],y0[j]])
#         plt.scatter(x_ini[0],x_ini[1],s=1,c='b')
#         print(f'x_ini={x_ini}')
#         sol = solve_system(max_step, (dim,f), 'RK45', 8, 0, x_ini)
#         plt.plot(sol.y[0],sol.y[1],label=f"x0={x_ini[0],x_ini[1]}")
# plt.scatter(1,1,s=10,c='r')
# plt.legend()
# plt.show()

##Heteroclinic orbits
u_2=np.array([0,1])
u_3=np.array([1,0])

u_ini=1e-6*u_2+1e-6*u_3

sol1=solve_system(max_step, (dim,f), 'RK45', 12, 0, u_ini)
plt.scatter(0,0,s=5,c='r')
plt.scatter(3,0,s=5,c='r')
plt.plot(sol1.y[0],sol1.y[1])
plt.show()

u_ini=1e-3*u_2+1e-6*u_3
sol2=solve_system(max_step, (dim,f), 'RK45', 12, 0, u_ini)
plt.plot(sol2.y[0],sol2.y[1])
plt.scatter(0,0,s=5,c='r')
plt.scatter(0,2,s=5,c='r')
plt.show()

