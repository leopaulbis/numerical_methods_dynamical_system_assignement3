import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

dim=2

#function of the system
#this function is function of R^n to R^n where n=dim
def f(x,a,b,c,d):
    return(np.array([a*x[0]+b*x[1],c*x[0]+d*x[1]]))

#define the system of ode
def system_ode(t,x,dim,g,a,b,c,d):
    y=np.zeros(dim) #contain all the values of x_1,x_2...etc
    derivative=np.zeros(dim)

    for i in range (dim): #set all the initial conditions
        y[i]=x[i]

    for i in range(dim):
        derivative[i]=f(x,a,b,c,d)[i]

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

#print(f(np.array([1,2]),1,2,1,1))


####Cas 1
###Portrait de phase
x=np.linspace(0,5,3)
y=np.linspace(0,5,3)
a=2
b=-5
c=1
d=-2

for i in range(len(x)):
    for j in range(len(y)):
        x0=[x[i],y[j]]
        print(x0)
        sol=solve_system(0.01,(dim,f,a,b,c,d),'RK45',8,0,x0)
        tab=[]

        # for k in range(len(sol.y[1])):
        #     if np.abs(sol.y[1][k])>70:
        #         tab.append(k)

        tab=np.array(tab)
        sol_y=np.array(sol.y[1])
        sol_x=np.array(sol.y[0])

        if len(tab)!=0:
            sol_y=np.delete(sol.y[1],tab)
            sol_x =np.delete(sol.y[0],tab)

        plt.plot(sol_x,sol_y)


x = np.linspace(-20, 20, 10)
y = np.linspace(-15, 15, 10)
X, Y = np.meshgrid(x, y)

# Calcul du champ vectoriel

U = a*X + b*Y
V = c*X + d*Y

# Normalisation du champ vectoriel pour des flèches de longueur uniforme
N = np.sqrt(U**2 + V**2)
U /= N
V /= N

# Tracer le portrait de phase avec des flèches
#plt.figure()
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=0.5)
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(-10, 10)
#plt.ylim(-10,10)

plt.show()

###Cas 2
##Portrait de phase
x=np.linspace(-5,5,3)
y=np.linspace(-5,5,3)
a=3
b=-2
c=4
d=-1

for i in range(len(x)):
    for j in range(len(y)):
        x0=[x[i],y[j]]
        print(x0)
        plt.scatter(x0[0], x0[1], s=5,c='red')
        sol=solve_system(0.01,(dim,f,a,b,c,d),'RK45',2,0,x0)
        tab=[]



        tab=np.array(tab)
        sol_y=np.array(sol.y[1])
        sol_x=np.array(sol.y[0])



        plt.plot(sol_x,sol_y)
#
x = np.linspace(-60, 60, 6)
y = np.linspace(-60, 60, 6)
X, Y = np.meshgrid(x, y)

# Calcul du champ vectoriel
a, b, c, d = 3, -2, 4, -1
U = a*X + b*Y
V = c*X + d*Y

# Normalisation du champ vectoriel pour des flèches de longueur uniforme
N = np.sqrt(U**2 + V**2)
U /= N
V /= N

# Tracer le portrait de phase avec des flèches
#plt.figure()
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=0.1)
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(-10, 10)
#plt.ylim(-10,10)

plt.show()



####Cas 3
###Portrait de phase
#il faut 14 portrait de phase
x=np.linspace(-3,3,5)
y=np.linspace(-3,3,5)
a=-1
b=0
b=0
c=3
d=2

for i in range(len(x)):
    for j in range(len(y)):
        x0=[x[i],y[j]]
        print(x0)
        sol=solve_system(0.01,(dim,f,a,b,c,d),'RK45',2,0,x0)
        tab=[]

        for k in range(len(sol.y[1])):
            if np.abs(sol.y[1][k])>70:
                tab.append(k)

        tab=np.array(tab)
        sol_y=np.array(sol.y[1])
        sol_x=np.array(sol.y[0])

        if len(tab)!=0:
            sol_y=np.delete(sol.y[1],tab)
            sol_x =np.delete(sol.y[0],tab)

        plt.plot(sol_x,sol_y)
#plt.show()


###computation of the unstable and stable manifolds
u_p=np.array([0,1]) #eigein vector for 2
u_m=np.array([1,-1]) #eigein vector for -1
##computation of the unstable manifolds (part +)


x0=1e-6*u_p
sol1=solve_system(0.01,(dim,f,a,b,c,d),'RK45',9,0,x0)
sol1_x=sol1.y[0]
sol1_y=sol1.y[1]

x0=-1e-6*u_p
sol2=solve_system(0.01,(dim,f,a,b,c,d),'RK45',9,0,x0)
sol2_x=sol2.y[0]
sol2_y=sol2.y[1]

sol_x=np.concatenate((sol1_x,sol2_x))
sol_y=np.concatenate((sol1_y,sol2_y))
print(sol_x)
print(sol_y)
plt.plot(sol_x,sol_y,'black',label="unstable manifolds")

##computation of the stable manifolds (part -)

x0=1e-6*u_m
sol1=solve_system(0.01,(dim,f,a,b,c,d),'RK45',-15,0,x0)
sol1_x=sol1.y[0]
sol1_y=sol1.y[1]

x0=-1e-6*u_m
sol2=solve_system(0.01,(dim,f,a,b,c,d),'RK45',-15,0,x0)
sol2_x=sol2.y[0]
sol2_y=sol2.y[1]

sol_x=np.concatenate((sol1_x,sol2_x))
sol_y=np.concatenate((sol1_y,sol2_y))
print(sol_x)
print(sol_y)
plt.plot(sol_x,sol_y,'b',label="stable manifold")
plt.legend()

plt.show()

