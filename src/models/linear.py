from sklearn.linear_model import LinearRegression

class BaselineLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        # Expects flattened 2D arrays: (N, num_features * seq_len)
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
