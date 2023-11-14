from math import sqrt
from ..tensor import Tensor
import numpy as np
from ..ops import Dot, AddBias

class Layer: pass

class Linear(Layer): 
    def __init__(self, in_features, out_features):
        stdv = sqrt(2. / in_features)  # He initialization standard deviation
        self.weights = Tensor(np.random.normal(0, stdv, (in_features, out_features)))
        self.bias = Tensor(np.zeros((1, out_features)))

    def __call__(self, a: Tensor) -> Tensor:
        o: Tensor = Dot().forward(a, self.weights)
        z: Tensor = AddBias().forward(o, self.bias)
        return z