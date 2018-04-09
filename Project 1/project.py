import numpy as np

class Architechture():
    def __init__(self, Input, Output, Samples, Alfa):

        self.input = Input
        self.output = Output

        # Retrieved from:
        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hicdden-layers-and-nodes-in-a-feedforward-neural-netw
        self.hidden = int(np.round(Samples / ((Input + Output) / Alfa)))

        self.hidden_weights = np.random.uniform(-0.5,0.5,[self.hidden, Input])
        self.output_weights = np.random.uniform(-0.5,0.5,[Output, self.hidden])

        self.hidden_bias = np.random.uniform(-0.5,0.5,[self.hidden, 1])
        self.output_bias = np.random.uniform(-0.5,0.5,[Output, 1])

        self.old_hidden_delta = 0
        self.old_output_delta = 0

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

    # from the input layer to hidden:
        # for each hidden neuron, calculate net, f(net) and df(net)/dnet
            # net = (input . W) + b
        # ==end for

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

    # from the hidden layer to output:
        # for each output neuron, calculate net, f(net) and df(net)/dnet
            # net = (f_h . W) + b
        # ==end for
    # ==end feed_forward }

    return Foward(f_hidden, f_output, df_hidden, df_output)

def backpropagation(X, Y, model, eta, threshold, momentum):

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

            # Update weights in the output layer
            # w'[i] = w[i] + (eta * (sum_j(delta_o[i][j] * f_h[i][j])))
            model.output_weights = np.asarray(model.output_weights + (eta * np.asmatrix((np.transpose(delta_output)) * np.asmatrix(fwd.f_hidden))) + momentum * model.old_hidden_delta)

            # Update bias in the output layer
            # b'[i] = b[i] + (eta * delta_o[i])
            model.output_bias = np.asarray(model.output_bias + (eta * np.asmatrix(np.transpose(delta_output))))

            # Update weights in the hidden layer
            # w'[i] = w[i] + (eta * (sum_j(delta_h[i][j] * x[i][j])))
            model.hidden_weights = np.asarray(model.hidden_weights + (eta * (np.transpose(np.asmatrix(delta_hidden)) * np.asmatrix(x))) + momentum * model.old_output_delta)

            # Update bias in the hidden layer
            # b'[i] = b[i] + (eta * delta_h[i])
            model.hidden_bias = np.asarray(model.hidden_bias + (eta * np.transpose(np.asmatrix(delta_hidden))))

            model.old_hidden_delta = delta_hidden
            model.old_output_delta = delta_output

        squaredError = squaredError / len(X)
        print ("Avarage squared error: ", squaredError)

    return model

def bluetooth(parameters):
    
    x1, y1, x2, y2 = read_bluetooth(parameters[0])
    xp, yp = process_bluetooth(x1, y1)
    nn = train_bluetooth(xp, yp, parameters[1], parameters[2],parameters[3])
    xp, yp = process_bluetooth(x2, y2)
    test_bluetooth(nn, xp, yp)

def read_bluetooth(percentage):
    
    dataset = np.loadtxt('bluetooth.csv', delimiter=',', skiprows=1)

    np.random.shuffle(dataset)
    
    q = int(dataset.shape[0] * percentage) + 2
    
    X_training = dataset[0:q, 0:3]
    Y_training = dataset[0:q, 3]
    
    X_test = dataset[q:80, 0:3]
    Y_test = dataset[q:80, 3]
    
    return X_training, Y_training, X_test, Y_test

def process_bluetooth(data, classes):
        
    p_data = np.zeros((data.shape[0], data.shape[1]))
    p_classes = np.zeros(classes.shape[0])

    max_bt1 = np.amin(data[:,0])
    max_bt2 = np.amin(data[:,1])
    max_bt3 = np.amin(data[:,2])

    max_class = np.amax(classes)

    for n in range(len(data)):
            
        p_data[n, 0] = data[n,0] / max_bt1
        p_data[n, 1] = data[n,1] / max_bt2
        p_data[n, 2] = data[n,2] / max_bt3

        p_classes[n] = classes[n] / max_class

    return p_data, p_classes

def train_bluetooth(X, Y, eta, threshold, momentum):
    
    model = Architechture(X.shape[1], 1, 79, 1)
    trained_model = backpropagation(X, Y, model, eta, threshold, momentum)
    
    return trained_model

def test_bluetooth(model, X, Y):

    right = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        fwd = foward(model, x)

        # print ("Expected: ", np.round(y))
        # print ("Result: ", np.round(fwd.f_output[0]) * np.round(y))
        # print
        
        if np.round(fwd.f_output[0] * np.round(y)) == np.round(y):
            right += 1

    print ("Accuracy(%): ", (right * 100) / len(X))

def test(parameters):
    
    x1, y1, x2, y2 = read_test(parameters[0])
    xp, yp = process_test(x1, y1)
    nn = train_test(xp, yp, parameters[1], parameters[2],parameters[3])
    xp, yp = process_test(x2, y2)
    test_test(nn, xp, yp)

def read_test(percentage):
    
    dataset = np.loadtxt('test.txt', delimiter=',', skiprows=1)
    
    q = int(dataset.shape[0] * percentage) + 2
    
    X_training = dataset[0:q, 0:14]
    Y_training = dataset[0:q, 14]
    
    X_test = dataset[q:464, 0:14]
    Y_test = dataset[q:464, 14]
    
    return X_training, Y_training, X_test, Y_test

def process_test(data, classes):
        
    p_data = np.zeros((data.shape[0], 14))
    p_classes = np.zeros(classes.shape[0])

    max_bt1  = np.amax(data[:,0])
    max_bt2  = np.amax(data[:,1])
    max_bt3  = np.amax(data[:,2])
    max_bt4  = np.amax(data[:,3])
    max_bt5  = np.amax(data[:,4])
    max_bt6  = np.amax(data[:,5])
    max_bt7  = np.amax(data[:,6])
    max_bt8  = np.amax(data[:,7])
    max_bt9  = np.amax(data[:,8])
    max_bt10  = np.amax(data[:,9])
    max_bt11 = np.amax(data[:,10])
    max_bt12 = np.amax(data[:,11])
    max_bt13 = np.amax(data[:,12])
    max_bt14 = np.amax(data[:,13])

    max_class = np.amax(classes)

    for n in range(len(data)):
            
        p_data[n, 0] = data[n, 0] / max_bt1
        p_data[n, 1] = data[n, 1] / max_bt2
        p_data[n, 2] = data[n, 2] / max_bt3
        p_data[n, 3] = data[n, 3] / max_bt4
        p_data[n, 4] = data[n, 4] / max_bt5
        p_data[n, 5] = data[n, 5] / max_bt6
        p_data[n, 6] = data[n, 6] / max_bt7
        p_data[n, 7] = data[n, 7] / max_bt8
        p_data[n, 8] = data[n, 8] / max_bt9
        p_data[n, 9] = data[n, 9] / max_bt10
        p_data[n, 10] = data[n, 10] / max_bt11
        p_data[n, 11] = data[n, 11] / max_bt12
        p_data[n, 12] = data[n, 12] / max_bt13
        p_data[n, 13] = data[n, 13] / max_bt14

        p_classes[n] = classes[n] / max_class

    return p_data, p_classes

def train_test(X, Y, eta, threshold, momentum):
    
    model = Architechture(X.shape[1], 1, 464, 5)
    trained_model = backpropagation(X, Y, model, eta, threshold, momentum)
    
    return trained_model

def test_test(model, X, Y):
    
    right = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        fwd = foward(model, x)

        # print ("Expected: ", y)
        # print ("Result: ", fwd.f_output)
        # print ()

        if np.round(fwd.f_output) == np.round(y):
            right += 1

    print ("Accuracy(%): ", (right * 100) / len(X))

def __main__():

    paramters = np.loadtxt('parameters.txt', delimiter=',', skiprows=1)

    for p in range(len(paramters)):
        bluetooth(paramters[p])

    for p in range(len(paramters)):
        test(paramters[p])

if __name__ == '__main__':
    
    __main__()