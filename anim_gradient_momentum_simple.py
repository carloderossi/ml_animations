import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function to optimize (Example: a simple quadratic function)
def f(x, y):
    return x**2 + y**2

# Gradient of the function
def gradient_f(x, y):
    return np.array([2*x, 2*y])

# Hyperparameters
# The choice of learning rate and momentum factor significantly impacts the performance.
momentum = 0.8    # 0.9 If the momentum factor (𝛽) is too high, it may cause oscillations, slowing down convergence
learning_rate = 0.01  # 0.05 Conversely, if the learning rate (𝜂) is not suitable, it might not utilize momentum effectively.
steps = 100
# Initialization: The starting point can influence the convergence rate. 
# If the initialization is such that the momentum method starts in a direction that does not immediately align well with the optimal path, 
# it may initially seem slower.
initial_position = np.array([2.0, 2.0])  # Starting point

# Gradient Descent without Momentum
positions_gd = [initial_position]
current_position = initial_position

for _ in range(steps):
    grad = gradient_f(current_position[0], current_position[1])
    current_position = current_position - learning_rate * grad
    positions_gd.append(current_position)

# Gradient Descent with Momentum
positions_mom = [initial_position]
current_position = initial_position
velocity = np.array([0.0, 0.0])

for _ in range(steps):
    grad = gradient_f(current_position[0], current_position[1])
    velocity = momentum * velocity - learning_rate * grad
    current_position = current_position + velocity
    positions_mom.append(current_position)

# Convert to numpy arrays for easier handling
positions_gd = np.array(positions_gd)
positions_mom = np.array(positions_mom)

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a grid of points
X = np.linspace(-2, 2, 100)
Y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Plot the surface with a color mesh
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# Plot initial points
point_gd, = ax.plot([], [], [], 'ro', label='Gradient Descent')
point_mom, = ax.plot([], [], [], 'bo', label='Gradient With Momentum')

annotation = ax.text2D(0.001, 0.99, f"Momentum factor: {momentum}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

def init():
    point_gd.set_data([], [])
    point_gd.set_3d_properties([])
    point_mom.set_data([], [])
    point_mom.set_3d_properties([])
    annotation.set_text('')
    return point_gd, point_mom, annotation 

def update(frame):
    point_gd.set_data([positions_gd[frame, 0]], [positions_gd[frame, 1]])
    point_gd.set_3d_properties([f(positions_gd[frame, 0], positions_gd[frame, 1])])

    point_mom.set_data([positions_mom[frame, 0]], [positions_mom[frame, 1]])
    point_mom.set_3d_properties([f(positions_mom[frame, 0], positions_mom[frame, 1])])
    
    #ax.text(0.25, 0.65, f"Step: {frame}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    for text in ax.texts:  # Remove all text objects from the axes
          text.remove()
    # ax.text2D(0.05, 0.95, f"{frame}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    #ax.text2D(0.05, 0.95, f"Step: {frame}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    #annotation.set_text(f"Step: {frame}")
    ax.text2D(0.001, 0.99, f"Momentum factor: {momentum}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

    return point_gd, point_mom

#ani = FuncAnimation(fig, update, frames=range(steps), init_func=init, interval=10, blit=False, cache_frame_data=True)
animation = FuncAnimation(fig, update, frames=range(steps), init_func=init, interval=30, blit=True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.get_current_fig_manager().set_window_title("Gradient Descent versus Gradient With Momentum")
plt.show()

# Save the animation
animation.save('gradient_descent_momemntum.gif', writer='pillow')
