import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algo import ValueFunctionWithApproximation

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.state_dims = state_dims

        # Input -> 32 -> 32 -> 1
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Fixed learning rate as given in the assignment
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
        )

        self.loss_fn = nn.MSELoss()

    def __call__(self,s):
        self.model.eval()
        state_tensor = torch.as_tensor(s, dtype=torch.float32).view(1, -1)

        with torch.no_grad():
            value = self.model(state_tensor)

        return float(value.item())

    def update(self,alpha,G,s_tau):
        self.model.train()

        self.optimizer.zero_grad()

        state_tensor = torch.as_tensor(s_tau, dtype=torch.float32).view(1, -1)
        target_tensor = torch.as_tensor([G], dtype=torch.float32)

        prediction = self.model(state_tensor).squeeze()
        loss = self.loss_fn(prediction, target_tensor)

        loss.backward()
        self.optimizer.step()

