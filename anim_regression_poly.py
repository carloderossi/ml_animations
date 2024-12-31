#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import r2_score
import animation_helper as ahlp
import calc_helper as chlp

# Generate synthetic data
x_train, y_train, x_test, y_test = chlp.gen_synt_data(42)

# Initialize figures and axes for two subplots
## fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Adjust figsize as needed
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Adjust figsize as needed

x_range = np.linspace(0, 10, 200).reshape(-1, 1)

# Plot for Polynomial Degree
train_scatter1 = ax1.scatter(x_train, y_train, color="blue", label="Training Data")
test_scatter1 = ax1.scatter(x_test, y_test, color="green", label="Testing Data")
# The method "plot" returns a tuple containing a list of Line2D objects: 
# we unpack the first (and only) element of that list into curve1.
curve1, = ax1.plot([], [], color="red", label="Model Curve")
ax1.legend()
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 60)
ax1.set_xlabel("Weight")
ax1.set_ylabel("Height")

# Plot for Bias-Variance Tradeoff
train_errors = []
test_errors = []
line_train, = ax2.plot([], [], label="Training Error", marker="o")
line_test, = ax2.plot([], [], label="Testing Error", marker="o")
ax2.set_xlabel("Polynomial Degree")
ax2.set_ylabel("Mean Squared Error")
ax2.set_title("Bias-Variance Tradeoff")
ax2.legend()
ax2.set_xlim(0, 15) # Set x-axis limit to accommodate 15 degrees
ax2.set_ylim(0, 50) # Set y-axis limit to accommodate errors

def update(degree):
    global train_errors, test_errors
    
    # Polynomial regression
    y_pred_train, y_pred_test, y_pred_range, train_error, test_error = chlp.polynomial_reg(degree, x_train, x_test, x_range, y_train, y_test)

    for text in ax1.texts:  # Remove all text objects from the axes
        text.remove()
    
    # Update plot title
    ax1.set_title(f"Polynomial Degree: {degree}")
    
    # Calculate adjusted R-squared
    n = len(y_test)  # Number of samples
    p = degree     # Number of predictors (polynomial degree)
    r2 = r2_score(y_test, y_pred_test)
    # Check if denominator will be zero and handle it
    if n - p - 1 <= 0:
        adjusted_r2 = np.nan  # or any other appropriate value to indicate an error
    else:
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    train_errors.append(train_error)
    test_errors.append(test_error)

    curve1.set_data(x_range, y_pred_range)

    # Update bias-variance plot
    line_train.set_data(range(1, len(train_errors) + 1), train_errors)
    line_test.set_data(range(1, len(test_errors) + 1), test_errors)

    # Display adjusted R-squared on the plot
    ax1.text(0.05, 0.85, f"Adjusted R-squared: {adjusted_r2:.4f}", transform=ax1.transAxes, fontsize=10, verticalalignment='top')

    # Update bias-variance plot
    line_train.set_data(range(1, len(train_errors) + 1), train_errors)
    line_test.set_data(range(1, len(test_errors) + 1), test_errors)
    # Update bias-variance plot (modified to ensure x-axis matches the polynomial degree)
    line_train.set_data(range(1, degree + 1), train_errors[:degree])  # Use degree to limit train errors
    line_test.set_data(range(1, degree + 1), test_errors[:degree])  # Use degree to limit test errors

    # ax2.set_xlim(1, degree)  # Adjust x-axis limit dynamically
    ax2.relim()            # Recalculate the data limits
    ax2.autoscale_view()    # Autoscale the view
   
    return curve1, line_train, line_test

# Create animation
ax1.grid(True)
ax2.grid(True)
#fig.suptitle("Polynomial Regression Animation")
ani = animation.FuncAnimation(fig, update, frames=range(1, 16), interval=1000, blit=False, cache_frame_data=True) # blit=False for updating text!!
ani.save("polynomial_regression.gif", writer="pillow")
plt.get_current_fig_manager().set_window_title("Polynomial Regression Animation") # ensure window's title
plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show()