import pandas as pd
import numpy as np
import adaptative_pca as apca
from sklearn.neural_network import MLPClassifier
from sklearn import decomposition


def readData():

    data = pd.read_csv(filepath_or_buffer='iris.data', header=None, sep=',')    
    
    shuffled = data.sample(frac=1)

    Y = shuffled.ix[:,4].values 
    # print ("Y: \n", Y)
    # print

    X = shuffled.ix[:,0:3].values
    # print ("X: \n", X)
    # print

    return X, Y

def multilayerPerceptron(fit_data, fit_class, predict_data, predict_class):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(fit_data, fit_class) 
    result = clf.predict(predict_data)

    # print(result)

    correct = 0

    for n in range(predict_data.shape[0]):

        if result[n] == predict_class[n]:
            correct += 1

    print('Accuracy(%): ', correct / predict_data.shape[0] * 100)

def classicPCA(X):
    
    pca = decomposition.PCA(n_components=4)
    pca.fit_transform(X)
    return pca.explained_variance_ratio_

def adaptativePCA(X):
    
    for i in range(20):
        pca = apca.Architechture(4, 4, 0.001, 0.001, 0.5, 1000, 0.0000001)
        pca.start(X)
        
    return pca.Y / np.sum(pca.Y) * 100    

def main():
    
    samples, labels = readData()
    
    variance1 = classicPCA(samples)
    variance2 = adaptativePCA(samples)
    
    a1 = np.argsort(variance1)
    a2 = np.argsort(variance2)

    # Debug
    # print(variance1)
    # print(variance2)
    # print(a1)
    # print(a2)
    # print(a1[-1], a1[-2])
    # print(a2[-1], a2[-2])
    # print(samples[0:105, [a1[-1],a1[-2]]].shape)
    # print(samples[105:, [a1[-1],a1[-2]]].shape)

    print("[Classic PCA]")
    multilayerPerceptron(samples[0:105, [a1[-1],a1[-2]]], labels[0:105], 
                         samples[105:,  [a1[-1],a1[-2]]], labels[105:])

    print("[Adaptative PCA Network]")
    multilayerPerceptron(samples[0:105, [a2[-1],a2[-2]]], labels[0:105], 
                         samples[105:,  [a2[-1],a2[-2]]], labels[105:])


if __name__ == '__main__':
    main()
    