from data import *
from nearest_neighbor import *

def nearest_neighbor(data):
    print("The accuracy of nearest neighbor, standardized nearest neighbor, and normalized nearest neighbor are as follows:\n")
    pred = nearest(data.X_train, data.y_train, data.X_test, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for basic data")
    print(f'\tThe accuracy of nearest neighbor {accuracy(pred, data.y_test):.3f}\n')
    std_pred = nearest(data.X_train_std, data.y_train, data.X_test_std, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for standardized data")
    print(f'\tThe accuracy of standardized nearest neighbor {accuracy(std_pred, data.y_test):.3f}\n')
    norm_pred = nearest(data.X_train_norm, data.y_train, data.X_test_norm, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for normalized data")
    print(f'\tThe accuracy of normalized nearest neighbor {accuracy(norm_pred, data.y_test):.3f}\n')
    std_norm_pred = nearest(data.X_train_std_norm, data.y_train, data.X_test_std_norm, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for normalized and standardized data")
    print(f'\tThe accuracy of normalized and standardized nearest neighbor {accuracy(std_norm_pred, data.y_test):.3f}\n')




if __name__ == '__main__':
    data = Data('training.csv', 'testing.csv')
    nearest_neighbor(data)