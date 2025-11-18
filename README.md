# eigen-pinns
Reserach on PINNs for eigenvalue problems

# Physics-Informed Neural Networks for Quantum Eigenvalue Problems
## Links
- https://arxiv.org/pdf/2203.00451
- https://github.com/henry1jin/quantumNN
## Key points
- Eigenvalue problem with Dirichlet boundary: $\mathcal{L} f(x) = \lambda f(x)$
- Inputs:
    - x
    - constant ones to learn $\lambda$
- Loss is split in 
    - orthogonal --> dot product of eigenfunctions
    - normalized --> very specific, check if only quantum mechanics applicable
    - regularization --> two terms
    - scanning mechanism --> ..is set to 0 --> so basically not used
- Symmetry embedded
    - check if only relevant for quantum
- Parametric trick
    - NN does not learn "full eigenfunctions"
    - instead we have $f(x, \lambda) = f_b + g(x) * NN(x, \lambda)$ to enforce boundary conditions


# Notes -> to-do
- check whether integral over the surface will be zero --> use 1.T @ M @ u --> worked on bunny
- plot eigenfunctions --> way too much noise --> smoothing did not help --> need different loss

DUMMY - ssh workflow stopped working -> need debugging





# Unsorted
https://arxiv.org/pdf/2208.13483 -- https://github.com/YangYuSCU/DE-PINN/tree/master
https://arxiv.org/pdf/2506.04375
https://arxiv.org/pdf/2211.04607 -- https://github.com/mariosmat/PINN_for_quantum_wavefunction_surfaces
https://arxiv.org/pdf/2010.05075 -- https://github.com/henry1jin/eigeNN
https://mediatum.ub.tum.de/doc/1632870/kwm0n4od0og42tg17pewgnx5s.pdf
https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_71.pdf
https://arxiv.org/pdf/2209.11134



https://github.com/benmoseley/harmonic-oscillator-pinn-workshop
https://github.com/314arhaam
