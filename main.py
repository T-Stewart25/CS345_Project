from data import *
from nearest_neighbor import *

def nearest(X_train, y_train, X_test, y_test):
    Knn = KNeighborsClassifier(350, algorithm='brute')
    Knn.fit(X_train, y_train)

    y_pred = Knn.predict(X_test)
   
    return y_pred
    
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


if __name__ == '__main__':
    data = Data('training.csv', 'testing.csv')
    print("The accuracy of nearest neighbor, standardized nearest neighbor, and normalized nearest neighbor are as follows:")
    pred = nearest(data.X_train, data.y_train, data.X_test, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for basic data")
    print(f'The accuracy of nearest neighbor {accuracy(pred, data.y_test):.3f}')
    std_pred = nearest(data.X_train_std, data.y_train, data.X_test_std, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for standardized data")
    print(f'The accuracy of standardized nearest neighbor {accuracy(std_pred, data.y_test):.3f}')
    norm_pred = nearest(data.X_train_norm, data.y_train, data.X_test_norm, data.y_test)  # Accessing attributes through the instance
    print("Training and testing data loaded successfully for normalized data")
    print(f'The accuracy of normalized nearest neighbor {accuracy(norm_pred, data.y_test):.3f}')

