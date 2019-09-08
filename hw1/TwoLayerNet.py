import torch.nn as nn


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

