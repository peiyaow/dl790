from keras.datasets import fashion_mnist
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# model parameters
# N: batch size
N, D_in, H1, H2, D_out = 64, 28*28, 1024, 1024, 10
lr = 1e-4
num_epochs = 20


class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# define model
class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(TwoLayerNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.Sigmoid(),
            nn.Linear(H1, H2),
            nn.Sigmoid(),
            nn.Linear(H2, D_out),
        )

    def forward(self, x):
        return self.model(x)


# loading data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# transform the data
n_train = x_train.shape[0]
n_test = x_test.shape[0]
x_train = x_train.reshape([n_train, 28*28])
x_test = x_test.reshape([n_test, 28*28])

ds = PrepareData(X=x_train, y=y_train)
ds = DataLoader(ds, batch_size=N, shuffle=True)

# dl model
model = TwoLayerNet(D_in, H1, H2, D_out)

# define loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for t in range(num_epochs):
    for x, y in ds:
        x = Variable(x).float()
        y = Variable(y).long()

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = loss_fn(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()









