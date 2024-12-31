import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.animation as animation

# Generate synthetic data
np.random.seed(42)
x_train = np.random.uniform(0, 10, 20)
y_train = 2 + 0.5 * x_train**2 - 0.3 * x_train + np.random.normal(0, 4, len(x_train))

x_test = np.random.uniform(0, 10, 20)
y_test = 2 + 0.5 * x_test**2 - 0.3 * x_test + np.random.normal(0, 4, len(x_test))

x_train = x_train[:, np.newaxis]
x_test = x_test[:, np.newaxis]

# Initialize figures and axes
fig, ax1 = plt.subplots(figsize=(8, 5))
x_range = np.linspace(0, 10, 200).reshape(-1, 1)

# Plot for Polynomial Degree
train_scatter1 = ax1.scatter(x_train, y_train, color="cyan", label="Training Data")
test_scatter1 = ax1.scatter(x_test, y_test, color="lime", label="Testing Data")
curve1, = ax1.plot([], [], color="red", label="Model Curve")
title = ax1.text(0.5, 1.05, '', transform=ax1.transAxes, ha='center', fontsize=12)
ax1.legend()
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 60)
ax1.set_xlabel("Weight")
ax1.set_ylabel("Height")

def update(degree):
    poly = PolynomialFeatures(degree=degree)
    title.set_text(f"Polynomial Degree: {degree}")
    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.transform(x_test)
    x_poly_range = poly.transform(x_range)

    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    y_pred_train = model.predict(x_poly_train)
    y_pred_test = model.predict(x_poly_test)
    y_pred_range = model.predict(x_poly_range)

    curve1.set_data(x_range, y_pred_range)
    title.set_text(f"Polynomial Degree: {degree}")
    
    return curve1, title

ani = animation.FuncAnimation(fig, update, frames=range(1, 16), interval=1000, blit=False)
ani.save("first_plot_animation.gif", writer="pillow")
plt.tight_layout()
plt.grid(True)
plt.get_current_fig_manager().set_window_title("Polynomial Regression Animation")
plt.show()
