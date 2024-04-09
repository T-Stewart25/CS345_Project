# Description: This file contains the implementation of the nearest neighbor classifier.
import numpy as np
import torch

class nearest_neighbor:
    def __init__(self):
        pass
    def fit(self, X, y):
        """train a nearest neighbor classifier.  Nothing much to do!"""
        self.X = X
        self.y = y
    def get_nearest(self, x):
        """returns the index of the training example closest to x"""
        distances = [distance(x, self.X[i]) 
                     for i in range(len(self.X))]
        return np.argmin(distances)
    def predict(self, x) :
        return self.y[self.get_nearest(x)]


#GPU ACCERATED  
class nearest_neighbor_torch:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        """Train a nearest neighbor classifier. Nothing much to do!"""
        self.X = torch.tensor(X)  # Convert to PyTorch tensor
        self.y = torch.tensor(y)  # Convert to PyTorch tensor
    
    def get_nearest(self, x):
        """Returns the index of the training example closest to x"""
        distances = torch.stack([torch.norm(x - self.X[i]) 
                                 for i in range(len(self.X))])
        return torch.argmin(distances)
    
    def predict(self, x):
        x_tensor = torch.tensor(x)  # Convert input to PyTorch tensor
        nearest_index = self.get_nearest(x_tensor)
        return self.y[nearest_index].item()