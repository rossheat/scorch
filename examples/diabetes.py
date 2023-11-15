import scorch
from scorch import nn
from scorch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes

class Diabetes(Dataset):
    def __init__(self):
        diabetes = load_diabetes()
        self.features = diabetes.data
        self.labels = diabetes.target 

    def __len__(self): 
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = Diabetes()
dataloader = DataLoader(dataset, batch_size=32)

class NeuralNetwork(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(10, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits

model = NeuralNetwork()
optimizer = scorch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(10):

    for batch, (X, y) in enumerate(dataloader):
        # Forward pass 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 

        if batch % 100 == 0:
            print(f'Epoch {epoch+1}: MSE loss: {loss.item():.4f}')