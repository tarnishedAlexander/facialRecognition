from .SigmoidActivation import SigmoidActivation
from .ReluActivation import ReluFunction

class ActivationFactory:

    @staticmethod
    def getActivation(activationType):

        if activationType == "sigmoid":
            return SigmoidActivation()
        
        elif activationType == "relu":
            return ReluFunction()
        
        else:
            raise ValueError(f"Activation type '{activationType}' not supported")
