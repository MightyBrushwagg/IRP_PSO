import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pandas as pd

#np.random.seed(random.randint(1, 300))

coefficients = np.random.uniform(-10, 10, (19))
leading_coefficients = np.random.uniform(0.5, 10, (2))
trig_displacements = np.random.uniform(0, 3.141, (4))
 
def f(x,y):
    "Objective function"
    #return (x-3.14)**4 + (y-2.72)**2 + np.sin(3*x+1.41) + np.cos(4*y-1.73)
    #return x**2 + (y+1)**2 + 5*np.cos(1.5*x+1.5) - 3*np.cos(2*x-1.5) 
    global coefficients, leading_coefficients, trig_displacements
    x_sum = leading_coefficients[0]*x**8 + coefficients[0]*x**7 + coefficients[1]*x**6 + coefficients[2]*x**5 + coefficients[3]*x**4 + coefficients[4]*x**3 + coefficients[5]*x**2 + coefficients[6]*x
    y_sum = leading_coefficients[1]*y**8 + coefficients[7]*y**7 + coefficients[8]*y**6 + coefficients[9]*y**5 + coefficients[10]*y**4 + coefficients[11]*y**3 + coefficients[12]*y**2 + coefficients[13]*y
    trig_sum = coefficients[14]*np.sin(x+trig_displacements[0]) + coefficients[15]*np.sin(y+trig_displacements[1]) + coefficients[16]*np.cos(x+trig_displacements[2]) + coefficients[17]*np.cos(y+trig_displacements[3])
    return x_sum+y_sum+trig_sum+coefficients[18]
    
# Compute and plot the function in 3D within [0,5]x[0,5]
x, y = np.array(np.meshgrid(np.linspace(-20,20,10000), np.linspace(-20,20,10000)))
z = f(x, y)

# Find the global minimum
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]
 
# Hyper-parameter of the algorithm
def c1(i):
    return 2
    #return i*0.001 #increasing

def c2(i):
    return 2
    #return (-1*i*0.001+0.1) #decreasing
    #return (-1*i*0.001+0.1)

w = 0.8

# Create particles
n_particles = 25

X = np.random.uniform(-20, 20, (2, n_particles))
V = np.random.uniform(-2, 2, (2, n_particles))

# Initialize data
pbest = X
pbest_obj = f(X[0], X[1])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()
iteration_found = 0

def reset():
    global V, X, pbest, pbest_obj, gbest, gbest_obj, coefficients, leading_coefficients, trig_displacements, x_min, y_min, iteration_found
    X = np.random.uniform(-20, 20, (2, n_particles))
    V = np.random.uniform(-2, 2, (2, n_particles))

    coefficients = np.random.uniform(-10, 10, (19))
    leading_coefficients = np.random.uniform(0.5, 10, (2))
    trig_displacements = np.random.uniform(0, 3.141, (4))

    x, y = np.array(np.meshgrid(np.linspace(-20,20,5000), np.linspace(-20,20,5000)))
    z = f(x, y)

    # Find the global minimum
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]

    # Initialize data
    pbest = X
    pbest_obj = f(X[0], X[1])
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    iteration_found = 0

def update(i):
    #print(i)
    "Function to do one iteration of particle swarm optimization"
    
    global V, X, pbest, pbest_obj, gbest, gbest_obj, iteration_found
    # Update params
    r1, r2 = np.random.rand(2)
    
    V = w * V + c1(i)*r1*(pbest - X) + c2(i)*r2*(gbest.reshape(-1,1)-X)
    X = X + V
    obj = f(X[0], X[1])
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    if pbest_obj.min() < gbest_obj:
        gbest_obj = pbest_obj.min()
        iteration_found = i + 1

# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([0,5])
ax.set_ylim([0,5])
 
def animate(i):
    "Steps of PSO: algorithm update and show in plot"
    title = 'Iteration {:02d}'.format(i)
    # Update params
    update(i)
    # Set picture
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])
    gbest_plot.set_offsets(gbest.reshape(1,-1))
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot
 
#anim = FuncAnimation(fig, animate, frames=list(range(1,50)), interval=500, blit=False, repeat=True)
#anim.save("PSO.gif", dpi=120, writer="imagemagick")

def runExperiment(iterations):
    for i in range(iterations):
        update(i)

data = {
    "Swarm minimum": [],
    "global minimum": [],
    "iteration found": []
}

df = pd.DataFrame(data)

data_file = open("dataFile.csv")
for i in range(1000):
    print(i)
    reset()
    runExperiment(100)
    #print("PSO found best solution at f({})={} on iteration:{}".format(gbest, gbest_obj, iteration_found))
    #print("Global optimal at f({})={}".format([x_min,y_min], f(x_min,y_min)))
    #print(gbest_obj)
    df.loc[-1] = [gbest_obj, f(x_min,y_min), iteration_found]
    df.index = df.index + 1  
    df = df.sort_index()

#print(df)
df.to_csv("dataFile.csv")

data_file.close()
