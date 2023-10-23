import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Define the system of ODEs
def system_ode(t, u):
    x, y = u
    dxdt = x * (3 - x - 2 * y)
    dydt = y * (2 - x - y)
    return [dxdt, dydt]

# Define the time span for the simulation
t_span = (0, 10)

# Create a grid of values for x and y
x = np.linspace(0.01, 3, 20)
y = np.linspace(0.01, 3, 20)
X, Y = np.meshgrid(x, y)

# Compute the direction field using the ODEs
DX, DY = system_ode(0, [X, Y])

# Normalize the direction field vectors for better visualization
norm = np.sqrt(DX**2 + DY**2)
DX = DX / norm
DY = DY / norm

# Plot the direction field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, DX, DY, norm, cmap=plt.cm.autumn, pivot='mid')

# Simulate and plot several trajectories from different initial conditions
for x0 in np.linspace(0.1, 3, 10):
    for y0 in np.linspace(0.1, 3, 10):
        initial_conditions = [x0, y0]
        solution = solve_ivp(system_ode, t_span, initial_conditions, dense_output=True)
        t = np.linspace(*t_span, 100)
        xy = solution.sol(t)
        plt.plot(xy[0], xy[1], 'b', alpha=0.4)

# Set axis labels and plot title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Global Dynamics of the System')

plt.xlim(0, 3)
plt.ylim(0, 3)

plt.grid(True)
plt.show()
