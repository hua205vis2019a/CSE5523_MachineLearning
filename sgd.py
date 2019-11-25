import numpy as np
import sympy as sm
sigma, n, N, t = [.05, .3], [50, 100, 500, 1000], 400, 30

def generateTrain1(u):
    trainExample = []
    return trainExample

def generateTrain2(u):
    trainExample = []
    return trainExample

def generateTest(u):
    testExample = []
    return testExample

def generateTestU(sigma):
    negU = [np.random.normal(-0.25, sigma, 4).tolist() for _ in range(200)]
    posU = [np.random.normal(0.25, sigma, 4).tolist() for _ in range(200)]
    return negU + posU

def loss(w, x, y):
    return np.log(1 + np.exp(-y*sum(w[i]*x[i] for i in range(5))))

def sgd(scenario, n):
    w = [0, 0, 0, 0, 0]






