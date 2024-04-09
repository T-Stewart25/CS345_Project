import numpy as np


classes = {
    "Normal": "0",
    "Fuzzers": "1",
    "Analysis": "2",
    "Backdoor": "3",
    "DoS": "4",
    "Exploits": "5",
    "Generic": "6",
    "Reconnaissance": "7",
    "Shellcode": "8",
    "Worms": "9"
}

def standardization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)  # Skip header
        for line in file:
            temp = line.strip().split(',')
            temp.append(classes[temp[-2]])
            data.append(temp)
    return data

def split_data(data):
    X_data = []
    y_data = []
    for line in data:
        X_data.append(line[:-3])
        y_data.append(line[-1])
    return X_data, y_data

def preprocess_data_with_standardization(training_file, testing_file):
    training_data = load_data(training_file)
    testing_data = load_data(testing_file)
    
    X_train, y_train = split_data(training_data)
    X_test, y_test = split_data(testing_data)
    
    
    # Standardize features
    #X_train = standardization(X_train)
    #X_test = standardization(X_test)
    
    return X_train, y_train, X_test, y_test
