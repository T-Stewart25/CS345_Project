from data import *








if __name__ == '__main__':
    X_train_std, y_train_std, X_test_std, y_test_std = preprocess_data_with_standardization('training.csv', 'testing.csv')
    X_train_norm, y_train_norm, X_test_norm, y_test_norm = preprocess_data_with_normalization('training.csv', 'testing.csv')
    
    


