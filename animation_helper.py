import matplotlib.pyplot as plt

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