import numpy as np
sigma, n, N, t = [.05, .3], [50, 100, 500, 1000], 400, 30

def generateTrainU(sigma, n):
    negU = [np.random.normal(-0.25, sigma, 4).tolist() for _ in range(n/2)]
    posU = [np.random.normal(0.25, sigma, 4).tolist() for _ in range(n/2)]
    return negU + posU

def generateTestU(sigma):
    negU = [np.random.normal(-0.25, sigma, 4).tolist() for _ in range(200)]
    posU = [np.random.normal(0.25, sigma, 4).tolist() for _ in range(200)]
    return negU + posU

def loss(w, x, y):


