import numpy as np
import samplers
import mesh_helpers
import diagnostics
from multigrid_model import MultigridGNN
from config import PINNConfig


def main():
    config = PINNConfig.from_yaml()

    print("Loading mesh...")
    mesh = mesh_helpers.load_mesh(config.mesh_file, normalize=True)

    print("Preprocessing mesh data...")
    sampler = samplers.Sampler(config)
    sampler.preprocess_mesh(mesh)

    print("Training physics-informed multiresolution GNN...")
    solver = MultigridGNN(config)
    U_refined = solver.train_multiresolution(sampler)

    # Save eigenfunctions
    print("Saving predicted eigenvectors...")
    mesh_helpers.save_eigenfunctions(mesh, U_refined, config.n_modes, config.vtu_file)

    # Run comprehensive diagnostics with REFINED eigenvectors
    print("Run diagnostics...")
    diagnostics.comprehensive_diagnostics(U_refined, mesh, sampler, config)

    # Visualize the sampler results
    if config.do_extensive_visuals:
        sampler.visualize('sampler')

    return U_refined


if __name__ == "__main__":
    main()
