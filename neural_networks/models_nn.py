import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        self.layer_sizes = layer_sizes
        self.hidden_layers = len(layer_sizes) - 2

        self.network = self.layers()

    def forward(self, x):
        x = self.network(x)

        return x

    def layers(self):
        layers = nn.Sequential()
        for i in range(self.hidden_layers):
            layers.add_module(f'hidden{i}', nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layers.add_module(f'activation{i}', nn.ReLU())
        layers.add_module('output', nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))

        return layers


class LSTM(nn.Module):

    def __init__(self, D_in, n_hidden, D_out, seq_len, n_layers=1):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(D_in, self.n_hidden, batch_first=False)

        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, D_out)
        self.activation = nn.ReLU

    def init_hidden(self, batchsize):
        return (torch.zeros(self.n_layers, batchsize, self.n_hidden),
                torch.zeros(self.n_layers, batchsize, self.n_hidden))

    def forward(self, x):
        hidden_cell = self.init_hidden(x.shape[1])

        out, hidden_cell = self.lstm(x, hidden_cell)
        out = self.activation(out)
        out = self.fc1(out[-1, :, :])
        out = self.activation(out)
        out = self.fc2(out)

        return out

# https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
class AE(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, output_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
