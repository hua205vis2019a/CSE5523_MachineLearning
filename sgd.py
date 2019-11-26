import numpy as np
from statistics import mean, stdev
from math import sqrt
from random import shuffle


# dimension, mu, sigma, n, N, iteration, alpha
d, mu, sigmas, ns, N, trial, alpha = 5, [-.25, .25], [.05, .3], [50, 100, 500, 1000], 400, 30, 0.01


def generateU(sigma, n):
    """Generate n d-1 dimensional Gaussian vectors
    :param sigma: sigma of Gaussian distribution
    :param n: number of vectors('u')
    :return: Gaussian vectors('u')
    """
    # first n/2 vectors based on y = -1, mu = [-1/4, -1/4, -1/4, -1/4], sigma
    # last n/2 vectors based on y = 1, mu = [1/4, 1/4, 1/4, 1/4], sigma
    uset = [[-1] + np.random.normal(mu[0], sigma, d-1).tolist() for _ in range(n//2)] + \
           [[1] + np.random.normal(mu[1], sigma, d-1).tolist() for _ in range(n//2)]
    # shuffle the set to make them with label 1 (or -1) not grouping together
    shuffle(uset)
    return uset


def generateSet1(uset):
    """Euclidean projection of 'u' onto X for Scenario 1
    :param uset: Gaussian vectors('u')
    :return: train/test set
    """
    trainSet = []
    for each in uset:
        example = [each[0]]
        # Euclidean projection for Scenario 1
        # for each dimension k of the vector:
        # if -1 <= k <= 1, keep consistent
        # if k < -1, use -1 instead
        # if k > 1, use 1 instead
        for k in each[1:]:
            if -1 <= k <= 1: example.append(k)
            elif k < -1: example.append(-1)
            else: example.append(1)
        trainSet.append(example)
    return trainSet


def generateSet2(uset):
    """Euclidean projection of 'u' onto X for Scenario 2
    :param uset: Gaussian vectors('u')
    :return: train/test set
    """
    trainSet = []
    for each in uset:
        # Euclidean projection for Scenario 2
        # if the Euclidean norm of the vector is less and equal to 1,
        #                              in the convex, keep consistent
        # otherwise, shorten it to 1
        euclidean = sqrt(sum(each[i] ** 2 for i in range(1, d)))
        if euclidean <= 1: trainSet.append(each)
        else: trainSet.append([each[0]] + [each[i]/euclidean for i in range(1, d)])
    return trainSet


def lossfunc(w, x, y):
    """Logistic loss function: ln(1 + exp(-y<w,x>))
    :param w: Ws
    :param x: test set
    :param y: label
    :return: logistic loss
    """
    x = x + [1]
    return np.log(1 + np.exp(-y*sum(w[i]*x[i] for i in range(d))))


def errorfunc(w, x, y):
    """Binary classification error: 1(sign(<w,(x,1)>) != y)
    :param w: Ws
    :param x: test set
    :param y: label
    :return: binary classification error
    """
    x = x + [1]
    return 0 if sum(w[i]*x[i] for i in range(d))*y > 0 else 1


def euclidean1(temp):
    """Euclidean projection onto C for Scenario 1 (This is for sgd function)
    The same theory mentioned in function generateSet1
    :param temp: input vector
    :return: the Euclidean projection of the vector onto C for Scenario 1
    """
    w = []
    for i, k in enumerate(temp):
        if -1 <= k <= 1: w.append(temp[i])
        elif k < -1: w.append(-1)
        else: w.append(1)
    return w


def euclidean2(temp):
    """Euclidean projection onto C for Scenario 2 (This is for sgd function)
    The same theory mentioned in function generateSet2
    :param temp: input vector
    :return: the Euclidean projection of the vector onto C for Scenario 2
    """
    euclidean = sqrt(sum(temp[i] ** 2 for i in range(d)))
    if euclidean <= 1: return temp
    else: return [temp[i] / euclidean for i in range(d)]


def sgd(euclidean, trainSet, n):
    """Own version of SGD algorithm
    :param euclidean: euclidean function name
    :param trainSet: train set
    :param n: iteration time
    :return: Ws
    """
    # initiation of w, w set, Ws
    w, wset, Ws = [0 for _ in range(d)], [[0 for _ in range(d)]], []
    # T iteration
    for i in range(n):
        # generate z and y
        z = trainSet[i][1:] + [1]
        y = trainSet[i][0]
        # the parameter of partial derivative with respect to each dimension
        # divides the quantity of that dimension
        param = -y * np.exp(-y * sum(w[i]*z[i] for i in range(d))) \
                / (1 + np.exp(-y * sum(w[i]*z[i] for i in range(d))))
        # generate Gt
        G = [param * z[i] for i in range(d)]
        # update w
        temp = [w[i] - alpha * G[i] for i in range(d)]
        w = euclidean(temp)
        wset.append(w)
    # calculate Ws
    for i in range(d):
        setIndex = 0
        for each in wset:
            setIndex += each[i]
        Ws.append(setIndex/len(wset))
    return Ws


def test(w, testSet):
    """Test Ws with test set and output the logistic loss and classification error
    :param w: Ws
    :param testSet: test set
    :return: [logistic loss, binary classification error]
    """
    lossSet, errorSet = [], []
    for i, each in enumerate(testSet):
        x = each[1:]
        y = each[0]
        lossSet.append(lossfunc(w, x, y))
        errorSet.append(errorfunc(w, x, y))
    # calculate the mean of loss and error
    loss, error = mean(lossSet), mean(errorSet)
    return [loss, error]


if __name__ == "__main__":
    """The whole procedure of train/test set generation, SGD, test and print out
    """
    # sigma: 0.05, 0.3
    for sigma in sigmas:
        print()
        print("*" * 150)
        print()
        print("sigma: ", sigma)
        # generate Gaussian vector for test set
        testU = generateU(sigma, N)
        # generate test set for each scenario
        testSet1, testSet2 = generateSet1(testU), generateSet2(testU)
        loss_error_1set, loss_error_2set = [[], [], [], []], [[], [], [], []]
        # 30 trials
        for _ in range(trial):
            # generate Gaussian vector for train set
            trainU = generateU(sigma, ns[-1])
            # generate train set for each scenario
            trainSet1, trainSet2 = generateSet1(trainU), generateSet2(trainU)
            # n: 50, 100, 500, 1000
            for i in range(len(ns)):
                # run SGD algorithm
                Ws1 = sgd(euclidean1, trainSet1[:ns[i]], ns[i])
                Ws2 = sgd(euclidean2, trainSet2[:ns[i]], ns[i])
                # test Ws and output loss and error
                loss_error_1set[i].append(test(Ws1, testSet1))
                loss_error_2set[i].append(test(Ws2, testSet2))
        # for n = 50, 100, 500, 1000, calculate and print out:
        # Logistic loss(Mean, Std Dev, Min, Excess Risk) and Classification error(Mean, Std Dev)
        for i in range(len(ns)):
            meanloss1 = mean(each[0] for each in loss_error_1set[i])
            stdloss1 = stdev(each[0] for each in loss_error_1set[i])
            minloss1 = min(each[0] for each in loss_error_1set[i])
            risk1 = meanloss1 - minloss1
            meanerror1 = mean(each[1] for each in loss_error_1set[i])
            stderror1 = stdev(each[1] for each in loss_error_1set[i])

            meanloss2 = mean(each[0] for each in loss_error_2set[i])
            stdloss2 = stdev(each[0] for each in loss_error_2set[i])
            minloss2 = min(each[0] for each in loss_error_2set[i])
            risk2 = meanloss2 - minloss2
            meanerror2 = mean(each[1] for each in loss_error_2set[i])
            stderror2 = stdev(each[1] for each in loss_error_2set[i])

            print("-"*150)
            print("n = ", ns[i])
            print("Scenario 1")
            print("Logistic loss")
            print("Mean: ", meanloss1, "| Std Dev: ", stdloss1, "| Min: ", minloss1, "| Excess Risk: ", risk1)
            print("Classification error")
            print("Mean: ", meanerror1, "| Std Dev: ", stderror1)
            print()
            print("Scenario 2")
            print("Logistic loss")
            print("Mean: ", meanloss2, "| Std Dev: ", stdloss2, "| Min: ", minloss2, "| Excess Risk: ", risk2)
            print("Classification error")
            print("Mean: ", meanerror2, "| Std Dev: ", stderror2)
