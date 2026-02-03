import numpy as np
from scipy.stats import stats

class LinearRegression:
    def __init__(self):
        self.beta = None
        self.X = None
        self.y = None
        self.n = None
        self.d = None


    def fit(self, X, y):
        self.X = X
        self.y = y

        self.n = X.shape[0]
        self.d = X.shape[1]

        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        Xty = X.T @ y

        self.beta = XtX_inv @ Xty


    def predict(self, X):
        y_hat = X @ self.beta
        return y_hat
    
    def residuals(self):
        r = self.y - self.predict(self.X)
        return r
    
    def sse(self):
        r2 = np.square(self.residuals())
        sse = np.sum(r2)
        return sse
    
    def variance(self):
        variance =  self.sse() / (self.n - self.d - 1)
        return variance

    def std(self):
        std = np.sqrt(self.variance())
        return std
    
    def rmse(self):
        rmse = np.sqrt(self.sse() / self.n)
        return rmse