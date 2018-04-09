import numpy as np
import math
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.misc import toimage
from sklearn import preprocessing

class Architechture():
    def __init__(self, Width, Height, Samples, Alpha, Interations):

        self.width = Width
        self.height = Height
        self.alphaConst = Alpha
        self.alpha = 0
        self.radiusConst = max(Width, Height) / 2
        self.radius = 0
        self.timeConst = 0
        self.interations = Interations
        self.step = 0

        self.weights = np.random.uniform(-0.5,0.5,[self.width * self.height, Samples])

    def bestMatchingUnit(self, sample):
        
        dist = np.zeros(self.width * self.height)
        i = 0
        
        for neuron in self.weights:
            dist[i] = math.sqrt(sum((sample - neuron)**2)) 
            i += 1

        return dist.argmin()

    def updateWeights(self, X, BMU):
        
        for neuron in range(self.weights.shape[0]):

            dist = math.sqrt(sum((self.weights[BMU, :] - self.weights[neuron, :])**2))

            if(dist < (self.radius)):
                
                theta = math.exp(-(dist**2)/(2*(self.radius**2)))
                delta = theta * self.alpha * (X - self.weights[BMU, :])
                self.weights[neuron, :] += delta

    def learn(self, data):
        
        self.alpha = self.alphaConst
        self.radius = self.radiusConst
        self.step = 1

        data_std = preprocessing.scale(data)

        while(self.step <= self.interations):
            
            print("Step " + str(self.step) + " of " + str(self.interations))
            print("Learning Rate: ", self.alpha)
            np.random.shuffle(data_std)

            for sample in data_std:
                
                bmu = self.bestMatchingUnit(sample)

                self.updateWeights(sample,bmu)
        
            self.alpha = self.alphaConst * math.exp(-(self.step)/self.interations)
            self.timeConst = self.interations / math.log(self.radius)
            self.radius = self.radiusConst * math.exp(-(self.step)/self.timeConst)
            self.step += 1

def readData():

    dataFrame = pd.read_csv(filepath_or_buffer='vinhos.txt', header=None, sep=',')
    dataFrame.dropna(how="all", inplace=True)

    data = dataFrame.ix[:, 1:13].values

    return data

def main():
    
    data = readData()

    konohen = Architechture(10, 10, data.shape[1], 0.1, 5000)

    konohen.learn(data)

    #print(konohen.weights)

    map = konohen.weights.mean(axis = 1)
    map = map.reshape((10,10))
    # print(map)
    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.show()
    
if __name__ == '__main__':
    main()









  