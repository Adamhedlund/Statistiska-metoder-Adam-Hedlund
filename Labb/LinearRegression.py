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
        
        self.n = X.shape[0]
        X = np.column_stack((np.ones(self.n), X))

        self.X = X
        self.y = y
        self.d = X.shape[1] - 1
        self.df = self.n - self.d - 1

        XtX = X.T @ X
        rank = np.linalg.matrix_rank(XtX)
        p = XtX.shape[0]

        if rank < p:
            XtX_inv = np.linalg.pinv(XtX)
        else:
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
        if self.X is None:
            raise ValueError("Model is not fitted. Call fit(X, y) first).")
        
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

    def summary(self):
        return {
            "n": self.n,
            "d": self.d,
            "df": self.df,
            "R2": self.r2(),
            "SSE": self.sse(),
            "Variance": self.variance(),
            "RMSE": self.rmse(),
            "F-statistic": self.f_statistic(),
            "F p-value": self.f_p_value(),
        }
