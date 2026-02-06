import numpy as np
import scipy.stats as stats

class LinearRegression:
    def __init__(self):
        self.beta = None
        self.X = None
        self.y = None
        self.n = None
        self.d = None
        self.df = None


    def fit(self, X, y):
        X = np.asanyarray(X, dtype=float)
        y = np.asanyarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got ndim={X.ndim}")
        if y.ndim != 1:
            raise ValueError(f"y must have be a 1D array, got ndim={y.ndim}")
        
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.d = X.shape[1] - 1
        self.df = self.n - self.d - 1

        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        Xty = X.T @ y

        self.beta = XtX_inv @ Xty
        self.XtX_inv = XtX_inv


    def predict(self, X):
        y_hat = X @ self.beta
        return y_hat
    
    def residuals(self):
        r = self.y - self.predict(self.X)
        return r
    
    def sse(self):
        r_square = np.square(self.residuals())
        sse = np.sum(r_square)
        return sse
    
    def variance(self):
        variance =  self.sse() / (self.df)
        return variance

    def std(self):
        std = np.sqrt(self.variance())
        return std
    
    def rmse(self):
        rmse = np.sqrt(self.sse() / self.n)
        return rmse
    
    def syy(self):
        y = np.square(self.y - np.mean(self.y))
        syy = np.sum(y)
        return syy
        
    def ssr(self):
        ssr = self.syy() - self.sse()
        return ssr
    
    def r2(self):
        r2 = self.ssr() / self.syy()
        return r2
    
    def f_statistic(self):
        f_stats = (self.ssr() / self.d) / self.variance()
        return f_stats
    
    def f_p_value(self):
        f = self.f_statistic()
        p = stats.f.sf(f,self.d, self.df)
        return p
    
    def pearson(self):
        Xf = self.X[:, 1:]
        d = Xf.shape[1]

        corr = np.eye(d)

        for i in range(d):
            for j in range (i + 1, d):
                r, _ = stats.pearsonr(Xf[:, i], Xf[:, j])
                corr[i, j] = r
                corr[j, i] = r
        return corr

    def covariance_matrix(self):
        cov = self.variance() * self.XtX_inv
        return cov
    
    def t_statistic(self):
        cov = self.covariance_matrix()
        diagonally = np.sqrt(np.diag(cov))

        t_stats = self.beta / diagonally
        return t_stats

    def t_p_values(self, alpha=0.05):
        t_stats = self.t_statistic()
        
        p_values = 2 * stats.t.sf(np.abs(t_stats), self.df)
        return p_values
    
    def confidence_intervals(self, alpha=0.05):
        df = self.df
        cov = self.covariance_matrix()
        se = np.sqrt(np.diag(cov))

        t_crit = stats.t.ppf(1 - alpha / 2, df)
        lower = self.beta - t_crit * se
        upper = self.beta + t_crit * se
        return np.column_stack([lower, upper])



x = np.array([0,1,2,3], dtype=float)
y = np.array([2,5.2,7.6,11.1], dtype=float)

X = np.column_stack([np.ones_like(x),x])

model = LinearRegression()
model.fit(X,y)
print("beta:", model.beta)
print("predict:", model.predict(X))
print("residuals:", model.residuals())
print("sse:", model.sse())
print("variance:", model.variance())
print("std:", model.std())
print("rmse:", model.rmse())
print("r2:", model.r2())
print("f:", model.f_statistic())
print("p:", model.f_p_value())
print("beta:", model.beta)
print("t:", model.t_statistic())
print("p:", model.t_p_values())
ci = model.confidence_intervals(0.05)
print("beta:", model.beta)
print("CI:", ci)