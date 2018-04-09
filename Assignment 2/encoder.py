import numpy as np

class Architechture():
    def __init__(self, Size):

        self.input = Size
        self.output = Size

        self.hidden = int(np.round(np.log2(Size)))

        #print (self.hidden)

        self.hidden_weights = np.random.uniform(-0.5,0.5,[self.hidden, self.input])
        self.output_weights = np.random.uniform(-0.5,0.5,[self.output, self.hidden])

        self.hidden_bias = np.random.uniform(-0.5,0.5,[self.hidden, 1])
        self.output_bias = np.random.uniform(-0.5,0.5,[self.output, 1])

        # print ("Hidden layer weights:\n",self.hidden_weights)
        # print ()
        # print ("Hidden layer bias:\n",self.hidden_bias)
        # print ()
        # print ("Output layer weights:\n",self.output_weights)
        # print ()
        # print ("Output layer bias:\n",self.output_bias)

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

            # Update weights in the output layer
            # w'[i] = w[i] + (eta * (sum_j(delta_o[i][j] * f_h[i][j])))
            model.output_weights = np.asarray(model.output_weights + (eta * np.asmatrix((np.transpose(delta_output)) * np.asmatrix(fwd.f_hidden))))

            # Update bias in the output layer
            # b'[i] = b[i] + (eta * delta_o[i])
            model.output_bias = np.asarray(model.output_bias + (eta * np.asmatrix(np.transpose(delta_output))))

            # Update weights in the hidden layer
            # w'[i] = w[i] + (eta * (sum_j(delta_h[i][j] * x[i][j])))
            model.hidden_weights = np.asarray(model.hidden_weights + (eta * (np.transpose(np.asmatrix(delta_hidden)) * np.asmatrix(x))))

            # Update bias in the hidden layer
            # b'[i] = b[i] + (eta * delta_h[i])
            model.hidden_bias = np.asarray(model.hidden_bias + (eta * np.transpose(np.asmatrix(delta_hidden))))

        squaredError = squaredError / len(X)
        print ("Avarage squared error: ", squaredError)

    return model

def encoder(eta, threshold):

    dataset = np.loadtxt('id_10.dat', skiprows=0)
    # print (dataset)
    # print ()

    Y = dataset[:,:]
    # print ("Y: \n", Y)
    # print ()

    X = dataset[:,:]
    # print ("X: \n", X)
    # print ()

    network = Architechture(dataset.shape[1])
    trained_network = backpropagation(X, Y, network, eta, threshold)

    count = 0

    for i in range(len(X)):
        
        x = X[i]
        y = Y[i]

        fwd = foward(trained_network, x)

        print ()
        print ("Input:    ", x)
        print ("Expected: ", y)
        print ("Output:   ", np.round(fwd.f_output))
        print ()

        if y[i] == np.round(fwd.f_output[i]):
            count += 1;

    
    print ("Model Accuracy (%): ", (count/len(X)) * 100)
    print ()

    return trained_network

if __name__ == '__main__':
    net = encoder(0.5, 1e-2)
    
    # print ("Hidden layer weights:\n",net.hidden_weights)
    # print ()
    # print ("Hidden layer bias:\n",net.hidden_bias)
    # print ()
    # print ("Output layer weights:\n",net.output_weights)
    # print ()
    # print ("Output layer bias:\n",net.output_bias)
