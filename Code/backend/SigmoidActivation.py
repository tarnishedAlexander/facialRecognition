from .Activation import ActivationFunction
import numpy as np

class SigmoidActivation(ActivationFunction):

    def activate(self, Z):
        cache = Z
        A = 1 / (1 + np.exp(-Z))
        return A, cache
    
    def derivate(self, dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
    