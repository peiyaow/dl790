from keras.datasets import fashion_mnist
import numpy as np
import torch.optim as optim
import torch.nn as nn


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


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# transform the data
n_train = x_train.shape[0]
n_test = x_test.shape[0]
x_train = x_train.reshape([n_train, 28*28])
x_test = x_test.reshape([n_test, 28*28])

# model parameters
# N: batch size
N, D_in, H1, H2, D_out = 64, 28*28, 1024, 1024, 10
lr = 1e-4

# dl model
model = TwoLayerNet(D_in, H1, H2, D_out)

# define loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)








