import numpy as np
import sys

class Architechture():
    def __init__(self, Input, Output, Samples, Alfa):

        self.input = Input
        self.output = Output

        # Retrieved from:
        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hicdden-layers-and-nodes-in-a-feedforward-neural-netw
        self.hidden = int(np.round(Samples / ((Input + Output) / Alfa)))
        print(self.hidden)

        self.hidden_weights = np.random.uniform(-0.5,0.5,[self.hidden, Input])
        self.output_weights = np.random.uniform(-0.5,0.5,[Output, self.hidden])

        self.hidden_bias = np.random.uniform(-0.5,0.5,[self.hidden, 1])
        self.output_bias = np.random.uniform(-0.5,0.5,[Output, 1])

class Foward():
    def __init__(self, f_hidden, f_output, df_hidden, df_output):

        self.f_hidden = f_hidden
        self.f_output = f_output
        self.df_hidden = df_hidden
        self.df_output = df_output


def sigmoid(net):
    return (1.0 / (1.0 + np.exp(-net)))

def sigmoid_gradient(net):
    return (sigmoid(net) * (1.0 - sigmoid(net)))

def foward(model, x):

    f_hidden = np.zeros(model.hidden)
    df_hidden = np.zeros(model.hidden)

    for j in range(model.hidden):
        net = np.dot(x, model.hidden_weights[j]) + model.hidden_bias[j]
        f_hidden[j] = sigmoid(net)
        df_hidden[j] = sigmoid_gradient(net)

    f_output = np.zeros(model.output)
    df_output = np.zeros(model.output)

    for k in range(model.output):
        net = np.dot(f_hidden, model.output_weights[k]) + model.output_bias[k]
        f_output[k] = sigmoid(net)
        df_output[k] = sigmoid_gradient(net)

    return Foward(f_hidden, f_output, df_hidden, df_output)

def backpropagation(X, Y, model, eta, threshold):

    squaredError = 2 * threshold

    while (squaredError > threshold):

        squaredError = 0

        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            
            fwd = foward(model, x)

            delta = y - fwd.f_output

            squaredError = squaredError + np.sum(delta * delta)

            delta_output = np.asmatrix(np.multiply(delta, fwd.df_output))
            delta_hidden = np.multiply(delta_output * model.output_weights, fwd.df_hidden)

            model.output_weights = np.asarray(model.output_weights + (eta * np.asmatrix((np.transpose(delta_output)) * np.asmatrix(fwd.f_hidden))))

            model.output_bias = np.asarray(model.output_bias + (eta * np.asmatrix(np.transpose(delta_output))))

            model.hidden_weights = np.asarray(model.hidden_weights + (eta * (np.transpose(np.asmatrix(delta_hidden)) * np.asmatrix(x))))

            model.hidden_bias = np.asarray(model.hidden_bias + (eta * np.transpose(np.asmatrix(delta_hidden))))

        squaredError = squaredError / len(X)
        print ("Avarage squared error: ", squaredError)

    return model


def read_iris(percentage):
    
    dataset = np.loadtxt('iris.data', delimiter=',', skiprows=0)

    np.random.shuffle(dataset)
    
    q = int(dataset.shape[0] * percentage) + 2
    
    X_training = dataset[0:q, 0:4]
    Y_training = dataset[0:q, 4]
    
    X_test = dataset[q:150, 0:4]
    Y_test = dataset[q:150, 4]
    
    return X_training, Y_training, X_test, Y_test

def process_iris_data(data):
        
    p_data = np.zeros((data.shape[0], data.shape[1]))

    max_col1 = np.amax(data[:,0])
    max_col2 = np.amax(data[:,1])
    max_col3 = np.amax(data[:,2])
    max_col4 = np.amax(data[:,3])

    for n in range(len(data)):
            
        p_data[n, 0] = data[n,0] / max_col1
        p_data[n, 1] = data[n,1] / max_col2
        p_data[n, 2] = data[n,2] / max_col3
        p_data[n, 3] = data[n,3] / max_col4

    return p_data

def process_iris_labels(labels, operation):

    if operation == 0:
        
        p_labels = np.zeros((labels.shape[0], 3))

        for n in range(len(labels)):
            p_labels[n, int(labels[n])] = 1 

        return p_labels
    else:
        p_labels = np.argmax(labels, axis=1)
        return p_labels

def train_iris(X, Y, percentage, eta, threshold):
    
    samples = int(percentage * X.shape[0])

    model = Architechture(X.shape[1], 3, samples, 2)
    trained_model = backpropagation(X, Y, model, eta, threshold)
    
    return trained_model

def test_iris(model, X, Y):

    right = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        fwd = foward(model, x)

        print ("Expected: ", y)
        print ("Result: ", np.argmax(fwd.f_output))
        print ()

        if np.argmax(fwd.f_output) == y:
            right += 1

    print ("Accuracy(%): ", (right * 100) / len(X))

if __name__ == '__main__':
    
    # input params
    # percentage eta threshold
    parameters = (sys.argv)

    x1, y1, x2, y2 = read_iris(float(parameters[1]))
    xp = process_iris_data(x1)
    yp = process_iris_labels(y1, 0)
    nn = train_iris(xp, yp, float(parameters[1]), float(parameters[2]), float(parameters[3]))
    xp = process_iris_data(x2)
    test_iris(nn, xp, y2)