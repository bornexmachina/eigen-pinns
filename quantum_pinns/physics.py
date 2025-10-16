import torch


def compute_gradient(f, x):
    """
    One-dimensional stationary Schrödinger's equation has the second order derivative
    So we have to compute d/dx (d/dx f)
    This has to be done explicitly and will not be covered by the loss.backward() which will 
    actually compute the gradient of the loss with regards to the model parameters theta
    """
    grad_outputs = torch.ones_like(f, device=x.device)
    gradient = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=grad_outputs, create_graph=True)[0]
    return gradient


def compute_schroedinger_loss_residual_and_hamiltonian(x, psi, E, V):
    """
    Schrödingers loss is the mean of the residual. Hamiltonian is used for verification. 
    The main reason to couple the computations is to use only one autograd pass
    """
    psi_dx = compute_gradient(psi, x)
    psi_ddx = compute_gradient(psi_dx, x)

    residual = -0.5 * psi_ddx + (V - E) * psi
    schroeddinger_loss = (residual.pow(2)).mean()
    hamiltonian = -0.5 * psi_ddx + V * psi

    return schroeddinger_loss, residual, hamiltonian


def compute_potential(x):
    """
    Compute the finite well potential with smooth boundaries.
    
    Replaces hard heaviside step functions with smooth sigmoid approximations
    to enable gradient computation during backpropagation.
    
    Parameters:
        x: tensor of position values
    
    Returns:
        V: potential tensor
    """
    
    # Smooth approximation of step functions using sigmoid
    # Higher k values make the transition sharper (closer to true step)
    # k=10 provides good balance between smoothness and sharp boundaries
    
    k = 10.0  # Steepness parameter - increase for sharper walls
    wall_height = 20.0
    wall_position = 1.7
    
    # Smooth left wall: heaviside(-x - 1.7) ≈ 1 - sigmoid(k*(-x - 1.7))
    left_wall = 1.0 - torch.sigmoid(k * (-x - wall_position))
    
    # Smooth right wall: heaviside(x - 1.7) ≈ sigmoid(k*(x - 1.7))
    right_wall = torch.sigmoid(k * (x - wall_position))
    
    # Combine walls
    V = (left_wall + right_wall) * wall_height
    
    return V