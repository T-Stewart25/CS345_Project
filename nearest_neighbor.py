from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class nearest_neighbor:
    def __init__(self):
        pass
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    def predict(self, X_test) :
        y_pred = np.array([])

        for x in X_test:
            predicted_class = self.y[np.argmin(np.linalg.norm(x-self.X, axis=1))]
            y_pred = np.append(y_pred, predicted_class)
        return y_pred
    
    