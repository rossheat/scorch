import scorch
from scorch import nn
from scorch.utils.data import Dataset, DataLoader

class FashionMNIST(Dataset):
    def __init__(self, csv_path: str): 
        df = self.read_csv(csv_path)
        self.features = self.scale(df.iloc[:, 1:])
        self.labels = df.iloc[:, 0]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx, :]
        label = self.labels[idx]
        return features, label

data_path = "/Users/johndoe/Developer/deep_learning/scorch/examples/data/fashion_mnist"
train_dataset = FashionMNIST(data_path + "/train.csv")
test_dataset = FashionMNIST(data_path + "/test.csv")

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)

class NeuralNetwork(nn.Module): 
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = scorch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for X, y in dataloader:
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")