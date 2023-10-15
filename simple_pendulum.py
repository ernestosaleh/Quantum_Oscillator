# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Simple pendulum solution:
# The equations of motion of a simple pendulum in the abscence of any dissipative force is determined by
# θ'' + (m * g/l) sin(θ) = 0
# This can be converted into a system of two first-order differential equations
#     φ = dθ/dt
# dφ/dt = -(m*g/l) * sin(θ)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Libraries

import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1      # mass (kg)
g = 9.81   # gravity (kg/m^2)
l = 2      # length (m)
dt = 0.01  # time step
cycle = 10 # time (s) 
t = np.linspace(0, cycle, int(cycle/dt)) # time (s)
n = len(t) # Amount of iterations

# Initial conditions
# State at t=0
θ0 = np.pi/4 # Angle initial condition
φ0 = 0 # Angular velocity initial condition

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function definition
θ = np.zeros(n)
θ[0]=θ0
φ = np.zeros(n)
φ[0]=φ0

def runge_kutta_4(f1, f2, x0, y0, z0):
    """
    Solves the system of the two differential equations stated in the header comments.
    Returns a the state of the next iteration for y1, z1.
    """
    # First parameter
    k1 = f1(x0, y0, z0)
    l1 = f2(x0, y0, z0)
    # Second parameter
    k2 = f1(x0 + dt/2, y0 + dt*k1/2, z0 + dt*l1/2)
    l2 = f2(x0 + dt/2, y0 + dt*k1/2, z0 + dt*l1/2)
    # Third parameter
    k3 = f1(x0 + dt/2, y0 + dt*k2/2, z0 + dt*l2/2)
    l3 = f2(x0 + dt/2, y0 + dt*k2/2, z0 + dt*l2/2)
    # Fourth parameter
    k4 = f1(x0 + dt, y0 + dt*k3, z0 + dt*l3)
    l4 = f2(x0 + dt, y0 + dt*k3, z0 + dt*l3)

    #Declaring the state of the next iteration
    y1 = y0 + dt*(k1 + 2 * k2 + 2 * k3 + k4)/6
    z1 = z0 + dt*(l1 + 2 * l2 + 2 * l3 + l4)/6

    return y1, z1


# Definition of the functions

#dθ/dt = φ
f1 = lambda t, theta, phi: phi

#dφ/dt = -(m*g/l) * sin(θ)
f2 = lambda t, theta, phi: -np.sin(theta)

for i in range(n-1):
    θ_new, φ_new = runge_kutta_4(f1, f2, t[i], θ[i], φ[i])
    θ[i+1]=θ_new
    φ[i+1]=φ_new


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].plot(t, θ)
axes[1].plot(t, φ)

plt.show()
