import time
import copy
import numpy as np
import torch
import torch.optim as optim
from models import Quantum_NN
import utils
import physics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nn(xL, xR, boundary_condition, hidden_dim, epochs, num_grid_points, lr, minibatch_number=1):
    """
    Train a neural network to solve the finite well quantum problem.
    
    Args:
        xL, xR: Domain boundaries
        boundary_condition: Represents f_b in the paer
        hidden_dim: Dimension of the hidden layers
        epochs: Number of training iterations
        n_train: Number of training points
        lr: Learning rate
        minibatch_number: Number of minibatches per epoch
    """
    
    # ============ INITIALIZATION ============
    # Initialize network and move to GPU
    nn_model = Quantum_NN(hidden_dim).to(device)
    
    # Setup optimizer with standard Adam hyperparameters for stability
    betas = (0.999, 0.9999)
    optimizer = optim.Adam(nn_model.parameters(), lr=lr, betas=betas)
    
    # ============ LOSS TRACKING DICTIONARIES ============
    # Track different loss components throughout training
    loss_histories = {
        'total': [],
        'SE': [],           # Schrödinger equation loss
        'normalization': [], # Wavefunction normalization loss
        'orthogonality': [], # Orthogonality constraint loss
        'energy': [],       # Energy eigenvalue losses
    }
    
    energy_history = []     # Track computed energy eigenvalues
    eigenstate_dict = {}    # Store best models for each energy level
    
    # ============ ORTHOGONALITY STATE TRACKING ============
    # These variables track the progressive orthogonalization procedure
    orth_counter = 0        # Counts how many eigenstates have been found
    convergence_threshold = np.exp(-14)  # Threshold for reweighting
    
    # Initialize eigenstate storage: {energy_level: (model, loss, data)}
    for i in range(50):
        eigenstate_dict[i] = (None, 1e+20)
    
    best_model = None
    best_loss = 1e+20
    
    # ============ GRID AND DATA SETUP ============
    # Create initial training grid and reshape for network input
    grid = torch.linspace(xL, xR, num_grid_points).reshape(-1, 1)
    batch_size = num_grid_points // minibatch_number
    
    # ============ TRAINING LOOP ============
    start_time = time.time()
    internal_se_loss = []  # Track SE loss for convergence detection
    
    for epoch in range(epochs):
        # Perturb evaluation points slightly (improves generalization)
        # Keep first point fixed at t0 (boundary condition)
        perturbed_grid = utils.perturb_grid_points(grid, xL, xR, perturbation_factor=0.03*xR)
        shuffled_grid = perturbed_grid[torch.randperm(num_grid_points)]
        shuffled_grid.requires_grad = True
        
        batch_start, batch_end = 0, batch_size
        epoch_loss = 0.0
        
        # ============ MINIBATCH LOOP ============
        for batch_idx in range(minibatch_number):
            # Extract current minibatch
            points_batch = shuffled_grid[batch_start:batch_end].to(device)
            
            # ============ FORWARD PASS ============
            # Get network predictions: wavefunction (nn) and energy (En)
            nn_output, energy = nn_model(points_batch)
            energy = torch.abs(energy)  # Ensure positive energy
            energy_history.append(energy[0].item())
            
            # Compute parametric solution (enforces boundary conditions)
            
            psi = utils.parametric_trick(points_batch, xL, xR, boundary_condition, nn_model).to(device)
            potential = physics.compute_potential(points_batch)
            
            # Main physics loss: Schrödinger equation residual
            se_loss, residual, hamiltonian_psi = physics.compute_schroedinger_loss_residual_and_hamiltonian(points_batch, psi, energy, potential)
            internal_se_loss.append(se_loss.cpu().detach().numpy())
            
            total_loss = se_loss
            
            # ============ NORMALIZATION CONSTRAINT ============
            # Add penalty if wavefunction isn't properly normalized
            norm_squared = torch.dot(psi[:, 0], psi[:, 0])
            target_norm_squared = (num_grid_points / (xR - xL)) * 1.0
            normalization_loss = (target_norm_squared - torch.sqrt(norm_squared)) ** 2
            total_loss += normalization_loss
            
            # ============ CONVERGENCE MONITORING ============
            # Compute rolling average of SE loss gradient to detect plateaus
            window_size = 1000
            if len(internal_se_loss) >= window_size + 1:
                loss_slope = np.mean(
                    np.array(internal_se_loss[-window_size:]) - 
                    np.array(internal_se_loss[-window_size-1:-1])
                )
            elif len(internal_se_loss) > 1:
                loss_slope = np.mean(
                    np.array(internal_se_loss[1:]) - 
                    np.array(internal_se_loss[:-1])
                )
            else:
                loss_slope = 0.0
            
            # Print progress every 300 epochs
            if epoch % 300 == 0:
                print(f'Epoch {epoch} | Energy {energy_history[-1]:.6f} | '
                      f'Loss slope {loss_slope:.2e} | Orth count {orth_counter}')
            
            # ============ ORTHOGONALIZATION PROCEDURE ============
            # Progressively enforce orthogonality to previously found eigenstates
            # This enables the network to find multiple eigenvalues
            
            trigger_reweight = (
                loss_slope < convergence_threshold and 
                loss_slope > 0 and
                epoch > 1000  # Avoid early reweighting
            )
            
            # Fixed reweighting at epoch 15000
            if epoch == 15000:
                nn_model.apply(utils.initialize_weights)
                orth_counter += 1
                print(f'Epoch {epoch} [FIXED REWEIGHT] | Orth count {orth_counter}')
            
            # Adaptive reweighting when convergence detected
            elif trigger_reweight:
                if orth_counter < 3:
                    nn_model.apply(utils.initialize_weights)
                    orth_counter += 1
                    print(f'Epoch {epoch} [ADAPTIVE REWEIGHT] | Orth count {orth_counter}')
                elif orth_counter == 3:
                    nn_model.sym = False  # Break symmetry for next eigenstate
                    orth_counter += 1
                    print(f'Epoch {epoch} [SYMMETRY BREAK] | Orth count {orth_counter}')
                else:
                    orth_counter += 1
                    print(f'Epoch {epoch} [CONTINUE] | Orth count {orth_counter}')
            
            # ============ ORTHOGONALITY LOSS ============
            # Penalize overlap with previously found eigenstates
            ortho_loss = torch.tensor(0.0)
            
            if orth_counter == 1 and eigenstate_dict[0][0] is not None:
                # Orthogonal to first eigenstate
                psi_prev = utils.parametric_trick(points_batch, xL, xR, boundary_condition, eigenstate_dict[0][0])
                overlap = torch.dot(psi_prev[:, 0], psi[:, 0]) ** 2
                ortho_loss = torch.sqrt(overlap) / 25
                
            elif orth_counter >= 2 and eigenstate_dict[0][0] is not None:
                # Orthogonal to multiple previous eigenstates
                psi_states = [utils.parametric_trick(points_batch, xL, xR, boundary_condition, eigenstate_dict[i][0])
                    for i in range(min(orth_counter, 4))
                    if eigenstate_dict[i][0] is not None
                ]
                combined_overlap = torch.dot(
                    torch.sum(torch.cat(psi_states, dim=1), dim=1),
                    psi[:, 0]
                ) ** 2
                ortho_loss = torch.sqrt(combined_overlap) / 25
            
            total_loss += ortho_loss
            
            # ============ BACKWARD PASS AND OPTIMIZATION ============
            total_loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += total_loss.cpu().detach().item()
            
            # ============ BOOKKEEPING ============
            # Track losses and update best model
            loss_histories['total'].append(total_loss.item())
            loss_histories['SE'].append(se_loss.item())
            loss_histories['normalization'].append(normalization_loss.item())
            loss_histories['orthogonality'].append(ortho_loss.item() if ortho_loss != 0 else 0)
            
            batch_start += batch_size
            batch_end += batch_size
        
        # ============ POST-EPOCH UPDATES ============
        # Keep best model overall (lowest SE loss)
        if se_loss < best_loss:
            best_model = copy.deepcopy(nn_model)
            best_loss = se_loss
        
        # Store best model for each energy level bin
        energy_bin = int(abs(energy[0].item()) // 1)
        if energy_bin < 50 and se_loss < eigenstate_dict[energy_bin][1]:
            eigenstate_dict[energy_bin] = (copy.deepcopy(nn_model), se_loss, (points_batch, residual, hamiltonian_psi, psi))
    
    # ============ CLEANUP AND RETURN ============
    end_time = time.time()
    runtime = end_time - start_time
    
    return best_model, loss_histories, runtime, nn_model #, eigenstate_dict