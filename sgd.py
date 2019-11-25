import numpy as np
from statistics import mean
from math import sqrt


sigma, n, N, t = [.05, .3], [50, 100, 500, 1000], 400, 30

def generateU(sigma, n):
    negU = [np.random.normal(-0.25, sigma, 4).tolist() for _ in range(n/2)]
    posU = [np.random.normal(0.25, sigma, 4).tolist() for _ in range(n/2)]
    return negU + posU


def generateSet1(u):
    trainSet = []
    for each in u:
        example = []
        for k in each:
            if -1 <= k <= 1: example.append(k)
            elif k < -1: example.append(-1)
            else: example.append(1)
        trainSet.append(example)
    return trainSet


def generateSet2(u):
    trainSet = []
    for each in u:
        euclidean = sqrt(sum(each[i] ** 2 for i in range(4)))
        if euclidean <= 1: trainSet.append(each)
        else: trainSet.append([each[i]/euclidean for i in range(4)])
    return trainSet


def lossfunc(w, x, y):
    x = x + [1]
    return np.log(1 + np.exp(-y*sum(w[i]*x[i] for i in range(5))))


def errorfunc(w, x, y):
    x = x + [1]
    return 0 if sum(w[i]*x[i] for i in range(5))*y > 0 else 1


def sgd1(n):
    w = [0, 0, 0, 0, 0]


def test(w, testSet):
    lossSet, errorSet = [], []
    for i, each in enumerate(testSet):
        x = each + [1]
        y = -1 if i < 200 else 1
        lossSet.append(lossfunc(w, x, y))
        errorSet.append(errorfunc(w, x, y))
    loss, error = mean(lossSet), mean(errorSet)
    return [loss, error]


