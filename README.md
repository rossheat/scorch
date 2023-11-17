I made this library to gain a better understanding of how backpropagation is used to train a neural network.

## Features
- **Backpropagation:** Scorch supports the backpropagation algorithm, enabling models to learn from their training data.
- **Neural Network Foundation Blocks:** Define a neural network with core components like Linear, ReLU, and Flatten.
- **Loss Functions:** Implement loss functions such as CrossEntropyLoss and MSELoss.
- **Optimizers:** Use gradient descent optimization with the provided SGD optimizer.
- **Data Handling:** Load and batch datasets effectively for model training with Dataset and DataLoader.
  
### Examples

Scorch Neural Network definition mimics PyTorch, as shown in the side-by-side comparison:

#### PyTorch
```python
import torch
import torch.nn as nn

class PyTorchNetwork(nn.Module): 
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = PyTorchNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

#### Scorch
```python
import scorch
import scorch.nn as nn

class ScorchNetwork(nn.Module): 
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)

model = ScorchNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = scorch.optim.SGD(model.parameters(), lr=1e-3)
```

Please find training examples in the `examples` directory.

## Installation

Clone the repository and install:

```bash
git clone https://github.com/rossheat/scorch.git
cd scorch
python setup.py install
python examples/diabetes.py
```

## License

Scorch is MIT licensed, as found in the LICENSE file.

## Acknowledgments

Scorch is inspired by PyTorch and is intended strictly for educational purposes, honoring the original work of the creators and contributors of PyTorch.
