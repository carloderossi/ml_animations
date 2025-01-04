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

# Parameters
learning_rate = 0.1
momentum_1 = 0.8
momentum_2 = 0.9
learning_rate_3 = 0.05  # Different learning rate for the third plot
steps = 100
initial_position = np.array([2.0, 2.0])  # Starting point

# Gradient Descent without Momentum
positions_gd = [initial_position]
current_position = initial_position

for _ in range(steps):
    grad = gradient_f(current_position[0], current_position[1])
    current_position = current_position - learning_rate * grad
    positions_gd.append(current_position)

# Gradient Descent with Momentum (Plot 1)
positions_mom_1 = [initial_position]
current_position = initial_position
velocity = np.array([0.0, 0.0])

for _ in range(steps):
    grad = gradient_f(current_position[0], current_position[1])
    velocity = momentum_1 * velocity - learning_rate * grad
    current_position = current_position + velocity
    positions_mom_1.append(current_position)

# Gradient Descent with Momentum (Plot 2)
positions_mom_2 = [initial_position]
current_position = initial_position
velocity = np.array([0.0, 0.0])

for _ in range(steps):
    grad = gradient_f(current_position[0], current_position[1])
    velocity = momentum_2 * velocity - learning_rate * grad
    current_position = current_position + velocity
    positions_mom_2.append(current_position)

# Gradient Descent with Momentum (Plot 3) with different learning rate
positions_mom_3 = [initial_position]
current_position = initial_position
velocity = np.array([0.0, 0.0])

for _ in range(steps):
    grad = gradient_f(current_position[0], current_position[1])
    velocity = momentum_1 * velocity - learning_rate_3 * grad
    current_position = current_position + velocity
    positions_mom_3.append(current_position)

# Convert to numpy arrays for easier handling
positions_gd = np.array(positions_gd)
positions_mom_1 = np.array(positions_mom_1)
positions_mom_2 = np.array(positions_mom_2)
positions_mom_3 = np.array(positions_mom_3)

# Animation
fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 5))

# Create a grid of points
X = np.linspace(-2, 2, 100)
Y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Plot the surfaces with a color mesh
for ax in axs:
    ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', edgecolor='none')

# Plot initial points
point_gd_1, = axs[0].plot([], [], [], 'ro', label='Gradient Descent')
point_mom_1, = axs[0].plot([], [], [], 'bo', label='Momentum = 0.8')

point_gd_2, = axs[1].plot([], [], [], 'ro', label='Gradient Descent')
point_mom_2, = axs[1].plot([], [], [], 'bo', label='Momentum = 0.9 (Oscillations)')

point_gd_3, = axs[2].plot([], [], [], 'ro', label='Gradient Descent')
point_mom_3, = axs[2].plot([], [], [], 'bo', label='Momentum = 0.8, Learning Rate = 0.05')

def init():
    for point_gd, point_mom in [(point_gd_1, point_mom_1), (point_gd_2, point_mom_2), (point_gd_3, point_mom_3)]:
        point_gd.set_data([], [])
        point_gd.set_3d_properties([])
        point_mom.set_data([], [])
        point_mom.set_3d_properties([])
    return point_gd_1, point_mom_1, point_gd_2, point_mom_2, point_gd_3, point_mom_3

def update(frame):
    point_gd_1.set_data([positions_gd[frame, 0]], [positions_gd[frame, 1]])
    point_gd_1.set_3d_properties([f(positions_gd[frame, 0], positions_gd[frame, 1])])

    point_mom_1.set_data([positions_mom_1[frame, 0]], [positions_mom_1[frame, 1]])
    point_mom_1.set_3d_properties([f(positions_mom_1[frame, 0], positions_mom_1[frame, 1])])

    point_gd_2.set_data([positions_gd[frame, 0]], [positions_gd[frame, 1]])
    point_gd_2.set_3d_properties([f(positions_gd[frame, 0], positions_gd[frame, 1])])

    point_mom_2.set_data([positions_mom_2[frame, 0]], [positions_mom_2[frame, 1]])
    point_mom_2.set_3d_properties([f(positions_mom_2[frame, 0], positions_mom_2[frame, 1])])

    point_gd_3.set_data([positions_gd[frame, 0]], [positions_gd[frame, 1]])
    point_gd_3.set_3d_properties([f(positions_gd[frame, 0], positions_gd[frame, 1])])

    point_mom_3.set_data([positions_mom_3[frame, 0]], [positions_mom_3[frame, 1]])
    point_mom_3.set_3d_properties([f(positions_mom_3[frame, 0], positions_mom_3[frame, 1])])

    return point_gd_1, point_mom_1, point_gd_2, point_mom_2, point_gd_3, point_mom_3

ani = FuncAnimation(fig, update, frames=range(steps), init_func=init, blit=True)

# Set titles for the subplots
axs[0].set_title("Momentum = 0.8")
axs[1].set_title("Momentum = 0.9 (Oscillations)")
axs[2].set_title("Momentum = 0.8, Learning Rate = 0.05")

plt.legend()
plt.show()
