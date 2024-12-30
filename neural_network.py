from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, x_train):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.n_inputs = x_train.shape[1]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
