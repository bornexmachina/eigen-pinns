import numpy as np
import meshio
import pyvista as pv
import matplotlib.pyplot as plt


def voxel_downsampling(points, hierarchy_levels):
    """
    Voxel grid downsampling for nested hierarchical sampling.
    Args:
        points: (N, 3) array of 3D points
        hierarchy_levels: list of target point counts for each level
    Returns:
        indices_per_level: dict mapping level index to array of selected point indices
    """
    n_points = points.shape[0]
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    extent = max_bound - min_bound
    
    indices_per_level = {}
    
    for level_idx, target_count in enumerate(hierarchy_levels):
        if target_count >= n_points:
            indices_per_level[level_idx] = np.arange(n_points)
            continue
        
        # Iteratively adjust voxel size to hit target
        # Start with a conservative estimate (assuming ~50% occupancy)
        volume = np.prod(extent)
        voxel_size = (volume / (target_count * 2)) ** (1/3)
        
        best_indices = None
        best_diff = float('inf')
        
        # Try multiple voxel sizes to get close to target
        for scale in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
            current_voxel_size = voxel_size * scale
            
            # Compute voxel grid
            grid_dims = np.ceil(extent / current_voxel_size).astype(int) + 1
            voxel_indices = ((points - min_bound) / current_voxel_size).astype(int)
            voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)
            
            # Flatten to 1D
            voxel_ids = (voxel_indices[:, 0] * grid_dims[1] * grid_dims[2] + 
                        voxel_indices[:, 1] * grid_dims[2] + 
                        voxel_indices[:, 2])
            
            # Select one point per voxel (closest to center)
            unique_voxels = np.unique(voxel_ids)
            selected_indices = []
            
            for voxel_id in unique_voxels:
                mask = (voxel_ids == voxel_id)
                voxel_points_idx = np.where(mask)[0]
                
                # Compute voxel center
                voxel_3d = np.array([
                    voxel_id // (grid_dims[1] * grid_dims[2]),
                    (voxel_id // grid_dims[2]) % grid_dims[1],
                    voxel_id % grid_dims[2]
                ])
                voxel_center = min_bound + (voxel_3d + 0.5) * current_voxel_size
                
                # Find closest point to center
                voxel_points = points[voxel_points_idx]
                distances = np.linalg.norm(voxel_points - voxel_center, axis=1)
                closest_idx = voxel_points_idx[np.argmin(distances)]
                selected_indices.append(closest_idx)
            
            # Check if this is closer to target
            diff = abs(len(selected_indices) - target_count)
            if diff < best_diff:
                best_diff = diff
                best_indices = np.array(selected_indices)
            
            # If we're very close or over target, we can stop
            if len(selected_indices) >= target_count * 0.95:
                break
        
        indices_per_level[level_idx] = best_indices[:target_count] if len(best_indices) > target_count else best_indices
    
    return indices_per_level


def farthest_point_sampling(points, n_samples, seed=None):
    """
    Farthest Point Sampling (FPS) for nested hierarchical sampling.
    
    Args:
        points: (N, 3) array of 3D points
        n_samples: number of points to sample
        seed: random seed for initial point selection
    
    Returns:
        indices: (n_samples,) array of selected point indices
    """
    n_points = points.shape[0]
    if n_samples >= n_points:
        return np.arange(n_points)
    
    rng = np.random.default_rng(seed=seed)
    
    # Start with a random point
    selected_indices = [rng.integers(0, n_points)]
    distances = np.full(n_points, np.inf)
    
    for _ in range(n_samples - 1):
        # Update distances to nearest selected point
        last_selected = selected_indices[-1]
        dists_to_last = np.linalg.norm(points - points[last_selected], axis=1)
        distances = np.minimum(distances, dists_to_last)
        
        # Select point farthest from all selected points
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)
    
    return np.array(selected_indices)


def simplify_mesh_decimation(input_file, target_vertices, save_file=False, output_dir=None, output_prefix=None):
    """Simplify mesh using quadric decimation algorithm."""
    # Read input mesh
    mesh = meshio.read(input_file)
    vertices = mesh.points
    faces = mesh.cells_dict.get('triangle', mesh.cells_dict.get('polygon', None))
    
    if faces is None:
        raise ValueError("No triangle or polygon cells found in mesh")
    
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to PyVista mesh
    faces_pv = np.hstack([[3] + list(face) for face in faces])
    pv_mesh = pv.PolyData(vertices, faces_pv)
    
    simplified_meshes = []
    
    for target_v in target_vertices:
        print(f"\nSimplifying to ~{target_v} vertices...")
        
        # Calculate target reduction ratio
        reduction = 1.0 - (target_v / len(vertices))
        reduction = max(0.0, min(0.99, reduction))  # Clamp between 0 and 0.99
        
        print(f"  Target reduction: {reduction*100:.1f}%")
        
        # Apply decimation
        simplified = pv_mesh.decimate(
            reduction,
            volume_preservation=True
        )
        
        # Extract vertices and faces
        verts_simplified = simplified.points
        faces_simplified = simplified.faces.reshape(-1, 4)[:, 1:4]
        
        # Save mesh
        output_mesh = meshio.Mesh(points=verts_simplified, cells=[("triangle", faces_simplified)])

        if save_file:
            output_file = f"{output_dir}/{output_prefix}_{len(verts_simplified)}.obj"
            meshio.write(output_file, output_mesh)
            print(f"  Saved: {output_file} ({len(verts_simplified)} vertices, {len(faces_simplified)} faces)")

        simplified_meshes.append(output_mesh)
    
    return simplified_meshes


def visualize_decimation(input_mesh_file, output_prefix, target_vertices):

    simplified_meshes = simplify_mesh_decimation(input_mesh_file, target_vertices=target_vertices)

    print("\n" + "="*50)
    print("Summary:")
    for idx, simplified_mesh in enumerate(simplified_meshes):
        print(f"  {idx} simplified mesh: {len(simplified_mesh.points)} vertices")

    # Plot the simplified meshes
    print("\nGenerating plots...")
    n_meshes = len(simplified_meshes) + 1  # +1 for original
    fig = plt.figure(figsize=(5*n_meshes, 5))

    # Plot original mesh
    original_mesh = meshio.read(input_mesh_file)
    original_faces = original_mesh.cells_dict.get('triangle', original_mesh.cells_dict.get('polygon', None))
    ax = fig.add_subplot(1, n_meshes, 1, projection='3d')
    ax.plot_trisurf(original_mesh.points[:, 0], 
                    original_mesh.points[:, 1], 
                    original_mesh.points[:, 2],
                    triangles=original_faces,
                    cmap='viridis', 
                    alpha=0.8, 
                    edgecolor='none')
    ax.set_title(f'Original\n{len(original_mesh.points)} vertices')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=95, azim=-90)

    # Plot simplified meshes
    for idx, simplified_mesh in enumerate(simplified_mesh, start=2):
        simplified_faces = simplified_mesh.cells_dict.get('triangle', simplified_mesh.cells_dict.get('polygon', None))
        ax = fig.add_subplot(1, n_meshes, idx, projection='3d')
        ax.plot_trisurf(simplified_mesh.points[:, 0], 
                        simplified_mesh.points[:, 1], 
                        simplified_mesh.points[:, 2],
                        triangles=simplified_faces,
                        cmap='plasma', 
                        alpha=0.8, 
                        edgecolor='none')
        ax.set_title(f'Simplified\n{len(simplified_mesh.points)} vertices')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=95, azim=-90)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_prefix}_comparison.png")
    plt.show()