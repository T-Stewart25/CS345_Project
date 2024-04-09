from data import *
from nearest_neighbor import *

def nearest(X_train, y_train, X_test, y_test):
    nn = nearest_neighbor()
    nn.fit(X_train, y_train)
    y_pred = np.array([nn.predict(X_test[i]) for i in range(len(X_test))])
    return y_pred
    
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preprocess_data('training.csv', 'testing.csv')
    X_train_std, y_train_std, X_test_std, y_test_std = preprocess_data_with_standardization('training.csv', 'testing.csv')
    X_train_norm, y_train_norm, X_test_norm, y_test_norm = preprocess_data_with_normalization('training.csv', 'testing.csv')
    pred = nearest(X_train, y_train, X_test, y_test)
    print(f'The accuracy of nearest neighbor {accuracy(pred, y_test):.3f}')
    std_pred = nearest(X_train_std, y_train_std, X_test_std, y_test_std)
    print(f'The accuracy of standardized nearest neighbor {accuracy(std_pred, y_test_std):.3f}')
    norm_pred = nearest(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
    print(f'The accuracy of normalized nearest neighbor {accuracy(norm_pred, y_test_norm):.3f}')
    
    


