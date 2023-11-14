from .layers import Layer

class Sequential: 
    def __init__(self, *items):
        self.items = items 
    
    def __call__(self, x):
        a = self.items[0](x)
        for item in self.items[1:]: 
            a = item(a)
        return a 

class Module: 
    def train(self): 
        # TODO:
        # Dropout layers randomly drop units to prevent overfitting 
        # Batch normalization layers should use the current batch's mean and variance.
        pass
    def eval(self): 
        # TODO: 
        # Dropout layers stop dropping out units
        # Batch normalization layers use running (accumulated) mean and variance instead of the current batch's statistics.
        pass

    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        parameters = []
        for item in self.stack.items:
            if isinstance(item, Layer):
                parameters.extend([item.weights, item.bias])
        return parameters 