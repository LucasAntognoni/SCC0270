import numpy as np

def readData(data):
    
    dataset = np.loadtxt(data)    
    # print (dataset)
    # print ()

    classID = dataset.shape[1]
    # print ("classID: ", classID)
    # print ()

    Y = dataset[:,classID - 1]
    # print ("Y: \n", Y)
    # print ()

    X = dataset[:,0:classID - 1]
    # print ("X: \n", X)
    # print ()

    return X, Y


def net_input(X, weights):
    return np.dot(X, weights[1:]) + weights[0]


def predict(X, weights):
    return np.where(net_input(X, weights) >= 0.0, 1, -1)


def train(X, Y, eta, threshold):
    
    weights = np.random.uniform(-0.5,0.5,[X.shape[1] + 1])
    costs = []
    squaredError = 2 * threshold

    while squaredError > threshold:
        
        output = net_input(X, weights)
        errors = (Y - output)

        weights[1:] += eta * X.T.dot(errors)
        weights[0] += eta * errors.sum()

        cost = (errors **2).sum() / 2.0

        print ("Squared Error: ", cost)

        squaredError = cost

        costs.append(cost)

    return weights


def results(predicted, label):

    acc = 0

    for i in range(predicted.shape[0]):
        
        if predicted[i] == label[i]:
            acc += 1
    
    acc = (acc / predicted.shape[0]) * 100

    print ("Accuracy (%): ", acc)

if __name__ == '__main__':
    
    X, Y = readData('symbols_train.dat')
    W = train(X, Y, 0.001, 1e-2)
    
    D, C = readData('symbols_test.dat')
    R = predict(D, W)

    results(R, C)