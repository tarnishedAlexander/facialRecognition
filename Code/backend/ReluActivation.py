from .Activation import ActivationFunction
import numpy as np

class ReluFunction(ActivationFunction):

    def activate(self, Z):
        A = np.maximum(0, Z)
        cache = Z
        return A, cache
    
    def derivate(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z<=0] = 0
        return dZ
    