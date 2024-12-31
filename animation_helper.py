import matplotlib.pyplot as plt
import calc_helper as chlp

def init_regression_plot(ax1, x_train, y_train,x_test, y_test):
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
    ax1.grid(True)
    return curve1, train_scatter1, test_scatter1

def init_biasvariance_plot(ax2):
    line_train, = ax2.plot([], [], label="Training Error", marker="o")
    line_test, = ax2.plot([], [], label="Testing Error", marker="o")
    ax2.set_xlabel("Polynomial Degree")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title("Bias-Variance Tradeoff")
    ax2.legend()
    ax2.set_xlim(0, 15) # Set x-axis limit to accommodate 15 degrees
    ax2.set_ylim(0, 50) # Set y-axis limit to accommodate errors
    ax2.grid(True)
    return line_train, line_test

def plot_error_surface(ax1, m, b, M, B, Z, x, y):
    surf = ax1.plot_surface(M, B, Z, cmap='viridis', alpha=0.5, edgecolor='none') # viridis, coolwarm
    point = ax1.plot([m], [b], [chlp.compute_error(m, b, x, y)], 'ro')[0]
    ax1.set_xlabel('m')
    ax1.set_ylabel('b')
    ax1.set_zlabel('Error')
    ax1.set_title('Error = {:.2f}'.format(chlp.compute_error(m, b, x, y)))
    return point, surf

def init_linear_plot(ax2, x, y, m, b, fig):
    ax2.scatter(x, y, color='blue')
    line, = ax2.plot(x, m * x + b, 'r-')
    ax2.set_title(f'm = {m:.2f}, b = {b:.2f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_ylim(-10, 10)
    ax2.grid(True)
    fig.tight_layout()
    return line