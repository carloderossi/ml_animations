import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import animation_helper as ahlp
import calc_helper as chlp

# Generate sample data
x,y = chlp.gen_sample_data()

# Create a grid for the error surface
m_vals, b_vals, M, B, Z = chlp.create_grid(x, y)

# Initialize parameters
m, b = -12, -12
learning_rate = 0.05

# Set up the figure and axes
fig = plt.figure(figsize=(25, 10))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Plot the error surface
point, surf = ahlp.plot_error_surface(ax1, m, b, M, B, Z, x, y)

# Plot the data and initial line
line = ahlp.init_linear_plot(ax2, x, y, m, b, fig)

# Update function for animation
def update(frame):
    global m, b
    m, b = chlp.gradient_descent_step(m, b, x, y, learning_rate)
    error = chlp.compute_error(m, b, x, y)

    # plot previous points
    ax1.scatter([m], [b], [error], color='navy') # forestgreen

    # Update the 3D point
    point.set_data([m], [b])
    point.set_3d_properties([error])
    ax1.set_title('Error = {:.4f}'.format(error))

    # Update the line in the 2D plot
    line.set_ydata(m * x + b)
    ax2.set_title(f'm = {m:.4f}, b = {b:.4f}')

    return point, line

# Parameters
frames = 100
interval = 100

# Create the animation with easing
animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, cache_frame_data=True)

# Save the animation
animation.save('gradient_descent.gif', writer='pillow')
     



""" 
# Easing function for smooth speed transition
def easing(t, start, end):
    return start + (end - start) * t**2

# Update function for animation with easing
def update_with_easing(frame):
    global m, b
    t = frame / (frames - 1)  # Normalized time variable from 0 to 1
    eased_frame = easing(t, 0, frames - 1)

    # Perform the original update using eased_frame
    m, b = chlp.gradient_descent_step(m, b, x, y, learning_rate * t)
    error = chlp.compute_error(m, b, x, y)

    # plot previous points
    ax1.scatter([m], [b], [error], color='navy')

    # Update the 3D point
    point.set_data([m], [b])
    point.set_3d_properties([error])
    ax1.set_title('Error = {:.4f}'.format(error))

    # Update the line in the 2D plot
    line.set_ydata(m * x + b)
    ax2.set_title(f'm = {m:.4f}, b = {b:.4f}')

    return point, line 
    
    
    # Create animation with easing 
    anim = FuncAnimation(fig, update_with_easing, frames=frames, interval=interval, blit=False, cache_frame_data=True)
    """