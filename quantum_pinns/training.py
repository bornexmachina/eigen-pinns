import time
import copy
import numpy as np
import torch
import torch.optim as optim
from models import Quantum_NN
import utils
import physics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train the NN
def train_nn(t0, tf, x1, neurons, epochs, n_train, lr, minibatch_number=1):
    par2 = 0
    fc0 = Quantum_NN(neurons)
    fc0.cuda()
    fc1 = 0
    betas = [0.999, 0.9999]
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = []
    Llim = 1e+20
    En_loss_history = []
    boundary_loss_history = []
    nontriv_loss_history = []
    SE_loss_history = []
    Ennontriv_loss_history = []
    criteria_loss_history = []
    En_history = []
    prob_loss = []
    EWall_history = []
    orth_losses = []
    di = (None, 1e+20)
    dic = {}
    for i in range(50):
        dic[i] = di
    orth_counter = 0
    swith = False
    internal_SE_loss = []

    grid = torch.linspace(t0, tf, n_train).reshape(-1, 1)
    
    ## TRAINING ITERATION    
    TeP0 = time.time()
    walle = -1.5
    last_psi_L = 0
    for tt in range(epochs):
        # Perturbing the evaluation points & forcing t[0]=t0
        t = utils.perturb_grid_points(grid, t0, tf, perturbation_factor=0.03*tf)
            
        # BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        t_b = t[idx]
        t_b.requires_grad = True
        t_f = t[-1]
        t_f = t_f.reshape(-1, 1)
        t_f.requires_grad = True
        loss = 0.0

        for nbatch in range(minibatch_number):
            # batch time set
            t_mb = t_b[batch_start:batch_end].cuda()
            
            # Network solutions 
            nn, En = fc0(t_mb)
            En = torch.abs(En)

            En_history.append(En[0].data.tolist()[0])

            psi = utils.parametric_trick(t_mb, t0, tf, x1, fc0).cuda()
            Pot = physics.compute_potential(t_mb)
            Ltot, f_ret, H_psi = physics.compute_schroedinger_loss_residual_and_hamiltonian(t_mb, psi, En.cuda(), Pot)
            
            SE_loss_history.append(Ltot)
            internal_SE_loss.append(Ltot.cpu().detach().numpy())
            criteria_loss = Ltot
            
            Ltot += ((n_train/(tf-t0))*1.0-torch.sqrt(torch.dot(psi[:,0],psi[:,0]))).pow(2)
            
            window = 1000
            if len(internal_SE_loss) >= window+1:
                rm = np.mean(np.array(internal_SE_loss[-window:])-np.array(internal_SE_loss[-window-1:-1]))
            else:
                rm = np.mean(np.array(internal_SE_loss[1:])-np.array(internal_SE_loss[:-1]))
            
            if tt % 300 == 0:
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)

            exp_thresh = -14
            if tt == 1.5e4:
                fc0.apply(utils.initialize_weights)
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 1:
                fc0.apply(utils.initialize_weights)
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 2:
                fc0.symmetry = False  # Changed from fc0.sym to fc0.symmetry
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 3:
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)

            if orth_counter == 1:
                par2 = utils.parametric_trick(t_mb, t0, tf, x1, dic[0][0])
                ortho_loss = torch.sqrt(torch.dot(par2[:,0], psi[:,0]).pow(2))/25
                orth_losses.append(ortho_loss)
                Ltot += ortho_loss  
            elif orth_counter == 2 or orth_counter == 3:
                par2 = utils.parametric_trick(t_mb, t0, tf, x1, dic[0][0])
                par3 = utils.parametric_trick(t_mb, t0, tf, x1, dic[3][0])
                ortho_loss = torch.sqrt(torch.dot(par2[:,0]+par3[:,0], psi[:,0]).pow(2))/25
                orth_losses.append(ortho_loss)
                Ltot += ortho_loss
            elif orth_counter == 4:
                par2 = utils.parametric_trick(t_mb, t0, tf, x1, dic[0][0])
                par3 = utils.parametric_trick(t_mb, t0, tf, x1, dic[3][0])
                par4 = utils.parametric_trick(t_mb, t0, tf, x1, dic[1][0])
                ortho_loss = torch.sqrt(torch.dot(par2[:,0]+par3[:,0]+par4[:,0], psi[:,0]).pow(2))/25
                orth_losses.append(ortho_loss)
                Ltot += ortho_loss

            En_loss_history.append(torch.exp(-1*En+walle).mean())
            EWall_history.append(walle)
            
            nontriv_loss_history.append(((n_train/(tf-t0))*1.0-torch.sqrt(torch.dot(psi[:,0],psi[:,0]))).pow(2))
            Ennontriv_loss_history.append(1/En[0][0].pow(2))
            
            # OPTIMIZER
            Ltot.backward(retain_graph=False)
            optimizer.step()
            loss += Ltot.cpu().data.numpy()
            optimizer.zero_grad()

            batch_start += batch_size
            batch_end += batch_size

        # keep the loss function history
        Loss_history.append(loss)       

        # Keep the best model (lowest loss) by using a deep copy
        if criteria_loss < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = criteria_loss

        E_bin = abs(En[0].data.tolist()[0])//1  
        if criteria_loss < dic[E_bin][1]:
            dic[E_bin] = (copy.deepcopy(fc0), criteria_loss, (t_mb, f_ret, H_psi, psi))

    TePf = time.time()
    runTime = TePf - TeP0  
    loss_histories = (Loss_history, boundary_loss_history, nontriv_loss_history, SE_loss_history, 
                      Ennontriv_loss_history, En_loss_history, criteria_loss_history, fc0, 
                      En_history, EWall_history, dic, orth_losses)
    return fc1, loss_histories, runTime, fc0