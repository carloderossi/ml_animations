import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''Generate synthetic data'''
def gen_synt_data(rnd_seed=42):
    np.random.seed(rnd_seed)
    x_train = np.random.uniform(0, 10, 20)
    y_train = 2 + 0.6 * x_train**2 - 0.3 * x_train + np.random.normal(0, 4, len(x_train))

    x_test = np.random.uniform(0, 10, 20)
    y_test = 2 + 0.6 * x_test**2 - 0.3 * x_test + np.random.normal(0, 4, len(x_test))

    x_train = x_train[:, np.newaxis]
    x_test = x_test[:, np.newaxis]
    
    return x_train, y_train, x_test, y_test

def polynomial_reg(degree, x_train, x_test, x_range, y_train, y_test):
    # Polynomial regression
    poly = PolynomialFeatures(degree=degree)
    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.transform(x_test)
    x_poly_range = poly.transform(x_range)

    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    y_pred_train = model.predict(x_poly_train)
    y_pred_test = model.predict(x_poly_test)
    y_pred_range = model.predict(x_poly_range)

    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    
    return y_pred_train, y_pred_test, y_pred_range, train_error, test_error

def calc_rsquared(degree, y_test, y_pred_test):
    n = len(y_test)  # Number of samples
    p = degree     # Number of predictors (polynomial degree)
    r2 = r2_score(y_test, y_pred_test)
    # Check if denominator will be zero and handle it
    if n - p - 1 <= 0:
        adjusted_r2 = np.nan  # or any other appropriate value to indicate an error
    else:
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
    return adjusted_r2