# We'll implement a PyTorch PINN-style solver that learns the first k generalized eigenpairs of (K, M).
# The network maps node coordinates X (N x 3) -> k outputs per node, yielding A (N x k).
# We M-orthonormalize A to produce U = A (A^T M A)^{-1/2} and minimize trace(U^T K U).
# We'll compare with scipy.linalg.eigh for verification.
# This code runs here and prints results.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy import linalg

# Repro
np.random.seed(0)
torch.manual_seed(0)

# Problem size and modes
N = 80   # number of nodes / DOFs (kept small for demo)
k = 6    # number of modes to learn

# Create synthetic M (SPD) and K that share generalized eigenstructure.
# We'll generate random "M-orthonormal" eigenvectors V and eigenvalues
# then build K = M V Lambda V^T so that K u = lambda M u.
A_rand = np.random.randn(N, N)
M_np = A_rand @ A_rand.T + N * np.eye(N)  # SPD
# create M-inner-product orthonormal basis V via generalized eigen-decomp of random SPD matrix
# technique: diagonalize M and form appropriate transform
# but simpler: get any V and then orthonormalize with respect to M using Gram-Schmidt
V = np.random.randn(N, k)
# M-orthonormalize columns of V
def m_gram_schmidt(V, M):
    # V: N x k
    V = V.copy()
    for i in range(V.shape[1]):
        vi = V[:, i]
        # subtract projections onto previous
        for j in range(i):
            vj = V[:, j]
            coef = vj.T @ (M @ vi)
            vi = vi - coef * vj
        # normalize in M-norm
        norm = math.sqrt(max(vi.T @ (M @ vi), 1e-12))
        V[:, i] = vi / norm
    return V

V = m_gram_schmidt(V, M_np)  # N x k

# assign eigenvalues (in ascending order)
true_eigs = np.linspace(1.0, 3.0, k)  # first k eigenvalues
# build K such that K V = M V Lambda; extend to full K by adding random part orthogonal to V
Lambda = np.diag(true_eigs)
K_np = M_np @ V @ Lambda @ V.T  # This enforces K V = M V Lambda, but K is rank-k
# make K full-rank SPD by adding small M-weighted complement
# construct projector onto complement (w.r.t M) and add small random SPD there
# build M-orthonormal basis for complement Uc (N x (N-k))
# we'll extend V to full M-orthonormal basis via simple method
Q = np.random.randn(N, N)
# use eigh on (Q^T M Q) to get transform that yields M-orthonormal columns
# simpler: perform Gram-Schmidt on random matrix to full-rank
V_full = np.zeros((N, N))
V_full[:, :k] = V
V_full[:, k:] = m_gram_schmidt(np.random.randn(N, N-k), M_np)
# now ensure full M-orthonormality via orthonormalization against first k
V_full[:, :k] = V
V_full[:, k:] = m_gram_schmidt(V_full[:, k:], M_np)
# create diagonal eigenvalues for full space: put larger values for complement
rest_eigs = np.linspace(4.0, 10.0, N-k)
Lambda_full = np.diag(np.concatenate([true_eigs, rest_eigs]))
K_np = M_np @ V_full @ Lambda_full @ V_full.T
# ensure symmetry numerically
K_np = 0.5*(K_np + K_np.T)
M_np = 0.5*(M_np + M_np.T)

# Convert to torch tensors (double precision for better numerical stability)
device = torch.device("cpu")
K = torch.from_numpy(K_np).double().to(device)
M = torch.from_numpy(M_np).double().to(device)

# Node coordinates X (N x 3) -- dummy here, we don't use geometry heavily but network will take X as input
X = np.random.randn(N, 3).astype(np.float64)
X_torch = torch.from_numpy(X).double().to(device)

# Reference solution via scipy.linalg.eigh for the first k modes (lowest eigenvalues)
# eigh solves K v = lambda M v; returns ascending eigenvalues
w_all, V_all = linalg.eigh(K_np, M_np)
w_ref = w_all[:k]
V_ref = V_all[:, :k]  # note: these are M-orthonormal by construction from eigh

print("Reference eigenvalues (first k):", np.round(w_ref, 6))

# Build the neural network that maps coordinates -> k outputs per node
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=k, hidden=[64,64]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h, dtype=torch.double))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_dim, dtype=torch.double))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)  # returns (N, k)

# Instantiate model and optimizer
model = MLP().to(device)
# initialize final layer small
for name, p in model.named_parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Helper: given A (N x k), return M-orthonormalized U = A (A^T M A)^{-1/2}
def m_orthonormalize(A, M):
    # A: (N, k), M: (N, N)
    # compute B = A^T M A (k x k)
    B = A.T @ (M @ A)  # k x k
    # symmetrize B
    B = 0.5*(B + B.T)
    # compute inverse sqrt of B via eigendecomposition (k small)
    s, Q = torch.linalg.eigh(B)  # s are eigenvalues
    # regularize small eigenvalues
    s_clamped = torch.clamp(s, min=1e-12)
    inv_sqrt = Q @ torch.diag(1.0/torch.sqrt(s_clamped)) @ Q.T
    U = A @ inv_sqrt
    return U, B, s

# Training loop: minimize trace(U^T K U)
max_epochs = 2000
print_every = 200
loss_history = []

for epoch in range(1, max_epochs+1):
    model.train()
    optimizer.zero_grad()
    A = model(X_torch)  # N x k
    U, B, s = m_orthonormalize(A, M)  # U is M-orthonormal
    # compute objective: trace(U^T K U) -> equals sum of Rayleighs for columns
    UK = U.T @ (K @ U)
    # symmetrize for numerical safety
    UK = 0.5*(UK + UK.T)
    loss = torch.trace(UK)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % print_every == 0 or epoch==1:
        # compute approximate eigenvalues by Rayleigh on current U
        with torch.no_grad():
            # the diagonal entries of UK are the Rayleigh quotients for each column if columns are M-orthonormal
            approx_vals = torch.diag(UK).cpu().numpy()
        print(f"Epoch {epoch:4d}, loss(trace)={loss.item():.6f}, approx_vals={np.round(approx_vals,6)}")

# After training, extract learned U and compute eigenvalues by Rayleigh on U and compare to ref
model.eval()
with torch.no_grad():
    A_final = model(X_torch)
    U_final, B_final, s_final = m_orthonormalize(A_final, M)
    UK_final = U_final.T @ (K @ U_final)
    approx_eigs = np.round(torch.diag(UK_final).cpu().numpy(), 8)
    # For better comparison, we can compute Ritz values from subspace U by solving small generalized eigenproblem
    # (U^T K U) c = mu (U^T M U) c, but U^T M U = I so just eig of UK_final
    mu, Wsmall = np.linalg.eigh(UK_final.cpu().numpy())
    mu = np.real(mu)
    # sort
    idx = np.argsort(mu)
    mu = mu[idx]
    print("\nLearned Ritz values (from U^T K U):", np.round(mu, 8))
    print("Reference eigenvalues (first k):    ", np.round(w_ref, 8))

# Small report on M-orthonormality error
with torch.no_grad():
    Mproj = U_final.T @ (M @ U_final)
    print("\nM-orthonormality error (Frobenius norm of U^T M U - I):",
          torch.norm(Mproj - torch.eye(k, dtype=torch.double)).item())

# Done. Print final minimal diagnostics
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('loss (trace)')
plt.title('Training loss (log scale)')
plt.grid(True)
plt.show()
