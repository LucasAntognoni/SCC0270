import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Architechture():

    def __init__(self, pattern, output, eta, mi, alpha, iterations, epslon):

        self.pattern = pattern
        self.output = output
        self.eta = eta
        self.mi = mi
        self.alpha = alpha
        self.iterations = iterations
        self.epslon = epslon

        self.Y = np.zeros(self.output)
        self.weights = np.random.uniform(0, 0.01, [self.pattern, self.output])
        self.side_weights = np.random.uniform(0, 0.01, [self.output, self.output])
        self.side_weights -= np.tril(self.side_weights)

    def updateWeights(self, x):

        # delta[i][j]
        delta = self.eta * x * np.transpose(self.Y)
        self.weights += delta
        self.weights /= np.amax(self.weights)

    def updateSideWeights(self):

        for l in range(self.output):
            for j in range(self.output):
                if l < j:
                    self.side_weights[l, j] += - self.mi * self.Y[l] * self.Y[j]

    def updateParams(self):

        self.eta = max(self.alpha * self.eta, 0.0001)
        self.mi = max(self.alpha * self.mi, 0.0002)

    def start(self, data):

        data_std = preprocessing.scale(data)

        np.random.shuffle(data_std)

        epoch = 0

        while(epoch <= self.iterations):

            print("Step " + str(epoch) + " of " + str(self.iterations))
            
            for p in range(data_std.shape[0]):
                
                sample = data_std[p]

                for i in range(self.output):
                    self.Y[i] = sum(self.weights[:, i] * sample) + sum(self.side_weights[i, :] * self.Y)

            self.updateWeights(sample)
            self.updateSideWeights()
            self.updateParams()

            if np.sum((self.side_weights)) <= self.epslon:
                break

            epoch += 1