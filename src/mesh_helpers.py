import numpy as np
import meshio
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from Mesh import Mesh


def normalize_mesh(mesh):
    centroid = mesh.verts.mean(0)
    std_max = mesh.verts.std(0).max() + 1e-12
    verts_normalized = (mesh.verts - centroid) / std_max
    return Mesh(verts=verts_normalized, connectivity=mesh.connectivity)


def load_mesh(file, normalize=True):
    mesh = Mesh(file)
    if normalize:
        mesh = normalize_mesh(mesh)
    return mesh


def visualize_mesh(mesh, title='Mesh Visualization', highlight_indices=None):
    """Visualize mesh with vertices, optionally highlighting specific points."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    
    # Plot full mesh with transparency
    ax.plot_trisurf(mesh.verts[:, 0], mesh.verts[:, 1], mesh.verts[:, 2], 
                    triangles=mesh.connectivity, alpha=0.3)
    
    # Highlight specific points if provided
    if highlight_indices is not None:
        highlighted_verts = mesh.verts[highlight_indices]
        ax.scatter(highlighted_verts[:, 0], highlighted_verts[:, 1], highlighted_verts[:, 2], 
                   c='fuchsia', s=10, alpha=0.8, label=f'{len(highlight_indices)} selected points')
        ax.legend()
    
    ax.set_title(title)
    ax.view_init(elev=130, azim=-90)
    plt.show()


def save_eigenfunctions(mesh, U_pred, n_modes, vtu_file):    
    centroid = mesh.verts.mean(0)
    std_max = mesh.verts.std(0).max()

    verts_new = (mesh.verts - centroid)/std_max

    m = Mesh(verts=verts_new, connectivity=mesh.connectivity)
    cells = [('triangle', m.connectivity)]

    m_out = meshio.Mesh(m.verts, cells, point_data={f'v{i}': U_pred[:, i] for i in range(0, n_modes)})
    m_out.write(vtu_file)


def compute_stiffness_and_mass_matrices(mesh):
    K, M = mesh.computeLaplacian()
    return coo_matrix(K), coo_matrix(M)


def mesh_to_edge_index(mesh):
    """
    Convert mesh triangular connectivity to edge_index format for GNN.
    Returns edges in both directions (undirected graph).
    """
    import torch
    connectivity = mesh.connectivity  # (n_tris, 3) array of vertex indices
    
    edges = []
    # Each triangle has 3 edges: (v0,v1), (v1,v2), (v2,v0)
    for tri in connectivity:
        edges.append([tri[0], tri[1]])
        edges.append([tri[1], tri[2]])
        edges.append([tri[2], tri[0]])
        # Add reverse edges for undirected graph
        edges.append([tri[1], tri[0]])
        edges.append([tri[2], tri[1]])
        edges.append([tri[0], tri[2]])
    
    edges = np.array(edges).T  # Shape: (2, n_edges)
    
    # Remove duplicate edges
    edges = np.unique(edges, axis=1)
    
    return torch.LongTensor(edges)


def meshio_to_Mesh(meshio_mesh, normalize=True):
    mesh = Mesh(verts=meshio_mesh.points, connectivity=meshio_mesh.cells_dict['triangle'])
    if normalize:
        mesh = normalize_mesh(mesh)
    return mesh
