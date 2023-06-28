import torch.nn as nn


class print_layer(nn.Module):
    def __init__(self):
        super(print_layer, self).__init__()

    def forward(self, obs):
        print(obs.shape)
        return obs


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=128):
        super(DQN, self).__init__()
        layers = [nn.Linear(n_observations, n_hidden),
                  nn.ReLU(),
                  nn.Linear(n_hidden, n_hidden),
                  nn.ReLU(),
                  nn.Linear(n_hidden, n_actions)]
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)
