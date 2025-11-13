import numpy as np
from Mesh import Mesh
import matplotlib.pyplot as plt


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
