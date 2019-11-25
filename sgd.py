import numpy as np
from statistics import mean
from math import sqrt

sigma, n, N, t, alpha = [.05, .3], [50, 100, 500, 1000], 400, 30, 0.01

def generateU(sigma, n):
    return [np.random.normal(-0.25, sigma, 4).tolist() for _ in range(n//2)] + \
           [np.random.normal(0.25, sigma, 4).tolist() for _ in range(n//2)]


def generateSet1(uset):
    trainSet = []
    for each in uset:
        example = []
        for k in each:
            if -1 <= k <= 1: example.append(k)
            elif k < -1: example.append(-1)
            else: example.append(1)
        trainSet.append(example)
    return trainSet


def generateSet2(uset):
    trainSet = []
    for each in uset:
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


def euclidean1(temp):
    w = []
    for i, k in enumerate(temp):
        if -1 <= k <= 1: w.append(temp[i])
        elif k < -1: w.append(-1)
        else: w.append(1)
    return w


def euclidean2(temp):
    euclidean = sqrt(sum(temp[i] ** 2 for i in range(5)))
    if euclidean <= 1: return temp
    else: return [temp[i] / euclidean for i in range(5)]


def sgd(euclidean, trainSet, n):
    w, wset, Ws = [0, 0, 0, 0, 0], [[0, 0, 0, 0, 0]], []
    for i in range(n):
        z = trainSet[i] + [1]
        y = -1 if i < n/2 else 1
        param = -y * np.exp(-y * sum(w[i]*z[i] for i in range(5))) \
                / (1 + np.exp(-y * sum(w[i]*z[i] for i in range(5))))
        G = [param * z[i] for i in range(5)]
        temp = [w[i] - alpha * G[i] for i in range(5)]
        w = euclidean(temp)
        wset.append(w)
    for i in range(5):
        setIndex = 0
        for each in wset:
            setIndex += each[i]
        Ws.append(setIndex/len(wset))
    return Ws


def test(w, testSet):
    lossSet, errorSet = [], []
    for i, each in enumerate(testSet):
        x = each + [1]
        y = -1 if i < 200 else 1
        lossSet.append(lossfunc(w, x, y))
        errorSet.append(errorfunc(w, x, y))
    loss, error = mean(lossSet), mean(errorSet)
    return [loss, error]


# if __name__ == "__main__":
#     # print("#"*107)
#     # print(" "*47, "|          Logistic loss            |  Classification error")
#     # print("Scenario |  sigma  | n | N |  #trials  |  Mean  |  Std Dev  |  Min  |  Excess Risk  |   Mean   |   Std Dev")
#     #
#     for each in sigma:
#         print("sigma")