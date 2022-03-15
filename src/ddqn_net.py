import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, target_input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.target_input_dims = target_input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc1_target = nn.Linear(self.target_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, target_flag=False):
        if target_flag:
            x = F.relu(self.fc1_target(state))
        else:
            x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions
