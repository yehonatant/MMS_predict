import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(101, 202),

            nn.ReLU(),
            nn.Linear(202, 202),

            nn.ReLU(),
            nn.Linear(202, 150),

            nn.ReLU(),
            nn.Linear(150, 50),

            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        pred = self.layers(x).flatten()
        return pred