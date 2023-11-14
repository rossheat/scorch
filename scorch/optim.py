class SGD: 
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self): 
        for p in self.parameters:
            p.data = p.data - self.lr * p.grad
            
    def zero_grad(self): 
        for p in self.parameters: 
            p.grad = 0 