import numpy as np
from .ActivationFactory import ActivationFactory

class NeuralNetwork:
    def __init__(self, layerDims, learning_rate=0.0075, num_iterations=3000):
        self.layerDims = layerDims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = self.init_parameters(layerDims)

    def init_parameters(self, layerDim):
        np.random.seed(1)
        parameters = {}
        L = len(layerDim)
        
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layerDim[l], layerDim[l-1]) / np.sqrt(layerDim[l-1])
            parameters["b" + str(l)] = np.zeros((layerDim[l], 1))
        return parameters

    def forward(self, A, W, b):
        Z = W.dot(A) + b
        cache = (A, W, b)
        return Z, cache
    

    def act_forward(self, Aprev, W, b, act):
        
        Z, linear_cache = self.forward(Aprev, W, b)
        A, act_cache = ActivationFactory.getActivation(act).activate(Z)
        cache = (linear_cache, act_cache)
        return A, cache

    def deep_forward(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            Aprev = A
            A, cache = self.act_forward(Aprev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
            caches.append(cache)
        AL, cache = self.act_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
        caches.append(cache)
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)
        return cost

    def backward(self, dZ, cache):
        Aprev, W, b = cache
        m = Aprev.shape[1]
        dW = 1. / m * np.dot(dZ, Aprev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dAprev = np.dot(W.T, dZ)
        return dAprev, dW, db

    def act_backward(self, dA, cache, act):
        linear_cache, act_cache = cache
        dZ = ActivationFactory.getActivation(act).derivate(dA, act_cache)
        dAprev, dW, db = self.backward(dZ, linear_cache)
        return dAprev, dW, db

    def deep_backward(self, AL, Y, cache):
        grads = {}
        L = len(cache)
        Y = Y.reshape(AL.shape)
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = cache[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.act_backward(dAL, current_cache, "sigmoid")
        
        for l in reversed(range(L - 1)):
            current_cache = cache[l]
            grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.act_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        return grads

    def update_params(self, params, grads, learning_rate):
        L = len(params) // 2
        for l in range(L):
            params["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            params["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
        return params

    def predict(self, X, Y, parameters):
        m = X.shape[1]
        p = np.zeros((1, m))
        proba, caches = self.deep_forward(X, parameters)

        for l in range(proba.shape[1]):
            if proba[0, l] > 0.5:
                p[0, l] = 1
            else:
                p[0, l] = 0
                
        print("Accuracy: " + str(np.sum((p == Y) / m)))
        return p

    def LLayer_model(self, X, Y, layerDims, learning_rate=0.0075, iter=3000):
        np.random.seed(1)
        costs = []
        parameters = self.init_parameters(layerDims)
        for l in range(iter):
            AL, caches = self.deep_forward(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.deep_backward(AL, Y, caches)
            parameters = self.update_params(parameters, grads, learning_rate)
            if l % 100 == 0 or l == iter - 1:
                print("Cost after iteration {}: {}".format(l, np.squeeze(cost)))
            if l % 100 == 0 or l == iter:
                costs.append(cost)
        
        return parameters, costs
