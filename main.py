def load_data():
    training = []
    testing = []
    with open('./testing.csv', 'r') as file:
        for line in file:
            training.append(line.strip().split(','))
    with open('./training.csv', 'r') as file:
        for line in file:
            testing.append(line.strip().split(','))
    return training, testing

if __name__ == '__main__':
    training, testing = load_data()