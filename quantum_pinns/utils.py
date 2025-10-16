import torch
import torch.nn as nn


def perturb_grid_points(grid, xL, xR, perturbation_factor):
    """
    For the training, a batch of xi points in the interval [xL, xR] is selected as input. In every training iteration (epoch) the
    input points are perturbed by a Gaussian noise to prevent the network from learning the solutions only at fixed points.
    """
    delta = grid[1] - grid[2]

    noise = delta * torch.randn_like(grid) * perturbation_factor
    perturbed_grid = grid + noise

    perturbed_grid[perturbed_grid < xL] = 2*xL - perturbed_grid[perturbed_grid < xL]
    perturbed_grid[perturbed_grid > xR] = 2*xR - perturbed_grid[perturbed_grid > xR]

    perturbed_grid[0] = xL
    perturbed_grid[-1] = xR

    perturbed_grid.requires_grad = False

    return perturbed_grid


def parametric_function(x, xL, xR):
    """
    Selecting an appropriate parametric function g(x) is necessary for enforcing boundary conditions. 
    The following parametric equation enforces a f (xL) = f (xR) = 0 boundary conditions
    """
    return (1 - torch.exp(-(x - xL))) * (1 - torch.exp(-(x - xR)))


def parametric_trick(x, xL, xR, boundary_value, neural_net):
    """
    The predicted eigenfunctions are defined using a parametric trick, i.e are not directly computed by the neural network
    The neural network returns a tuple: network_prediction and learnable_lambda
    """
    network_prediction, _ = neural_net(x)
    g = parametric_function(x, xL, xR)
    return boundary_value + g * network_prediction


def initialize_weights(layer):
    if isinstance(layer, nn.Linear) and layer.weight.shape[0] != 1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias.data)