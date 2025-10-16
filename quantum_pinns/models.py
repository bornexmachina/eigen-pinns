import torch
import torch.nn as nn


class Sin(nn.Module):
  """
  Hamiltonian neural networks for solving differential equations https://arxiv.org/pdf/2001.11107
  The use of sin(·) instead of more common activation functions, such as Sigmoid(·) and tanh(·), 
  significantly accelerates the network's convergence to a solution
  """
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.sin(x)
  

class Quantum_NN(nn.Module):
    """
    Network for the 1D PINN task
    """
    def __init__(self, hidden_dim=10, symmetry=True):
        super().__init__()
        self.symmetry = symmetry

        self.activation = Sin()
        self.input = nn.Linear(1, 1)
        self.first_hidden = nn.Linear(2, hidden_dim)
        self.second_hidden = nn.Linear(hidden_dim + 1, hidden_dim)
        self.output = nn.Linear(hidden_dim + 1, 1)

    def forward(self, t):
        learnable_lambda = self.input(torch.ones_like(t))

        first_hidden_symmetric = self.activation(self.first_hidden(torch.cat((t, learnable_lambda), dim=1)))
        first_hidden_antisymmetric = self.activation(self.first_hidden(torch.cat((-t, learnable_lambda), dim=1)))

        secondd_hidden_symmetric = self.activation(self.second_hidden(torch.cat((first_hidden_symmetric, learnable_lambda), dim=1)))
        second_hidden_antisymmetric = self.activation(self.second_hidden(torch.cat((first_hidden_antisymmetric, learnable_lambda), dim=1)))

        if self.symmetry:
            network_prediction = self.output(torch.cat((secondd_hidden_symmetric + second_hidden_antisymmetric, learnable_lambda), dim=1))
        else:
            network_prediction = self.output(torch.cat((secondd_hidden_symmetric - second_hidden_antisymmetric, learnable_lambda), dim=1))

        return network_prediction, learnable_lambda
