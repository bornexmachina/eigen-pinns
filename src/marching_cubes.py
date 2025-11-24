import numpy as np
import meshio
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

def mesh_to_voxels(vertices, faces, resolution):
    """Convert mesh to voxel grid with signed distance field."""
    # Get bounding box with padding
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    padding = 0.1 * (max_bound - min_bound)
    min_bound -= padding
    max_bound += padding
    
    # Create voxel grid
    grid_shape = (resolution, resolution, resolution)
    x = np.linspace(min_bound[0], max_bound[0], resolution)
    y = np.linspace(min_bound[1], max_bound[1], resolution)
    z = np.linspace(min_bound[2], max_bound[2], resolution)
    
    # Create mesh grid
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
    
    # Compute signed distance field (simplified: just distance to nearest vertex)
    # For better results, use proper point-to-mesh distance
    voxel_grid = np.zeros(grid_shape)
    
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                point = np.array([x[i], y[j], z[k]])
                # Distance to nearest vertex (simple approximation)
                dist = np.min(np.linalg.norm(vertices - point, axis=1))
                voxel_grid[i, j, k] = dist
    
    # Make it signed (negative inside, positive outside)
    # Simple heuristic: if close to vertices, assume inside
    threshold = np.percentile(voxel_grid, 30)
    voxel_grid = np.where(voxel_grid < threshold, -voxel_grid, voxel_grid)
    
    return voxel_grid, (min_bound, max_bound)

def mesh_to_voxels_fast(vertices, faces, resolution):
    """Faster voxelization using rasterization approach."""
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    padding = 0.05 * (max_bound - min_bound)
    min_bound -= padding
    max_bound += padding
    
    # Initialize binary voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Voxelize by checking if voxel centers are inside mesh
    spacing = (max_bound - min_bound) / resolution
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        # Get bounding box of triangle in voxel space
        tri_min = ((np.min([v0, v1, v2], axis=0) - min_bound) / spacing).astype(int)
        tri_max = ((np.max([v0, v1, v2], axis=0) - min_bound) / spacing).astype(int)
        
        # Clamp to grid bounds
        tri_min = np.maximum(tri_min, 0)
        tri_max = np.minimum(tri_max, resolution - 1)
        
        # Mark voxels in bounding box
        for i in range(tri_min[0], tri_max[0] + 1):
            for j in range(tri_min[1], tri_max[1] + 1):
                for k in range(tri_min[2], tri_max[2] + 1):
                    voxel_grid[i, j, k] = True
    
    # Convert to signed distance field
    # Inside = negative, outside = positive
    sdf = distance_transform_edt(~voxel_grid) - distance_transform_edt(voxel_grid)
    
    return sdf, (min_bound, max_bound)

def coarsen_mesh_marching_cubes(input_file, output_prefix, target_vertices=[256, 512, 1024]):
    """Create coarser meshes using voxelization and marching cubes."""
    # Read input mesh
    mesh = meshio.read(input_file)
    vertices = mesh.points
    faces = mesh.cells_dict.get('triangle', mesh.cells_dict.get('polygon', None))
    
    if faces is None:
        raise ValueError("No triangle or polygon cells found in mesh")
    
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    results = []
    
    for target_v in target_vertices:
        # Estimate resolution needed (rough heuristic)
        # Marching cubes typically produces vertices ~ resolution^2
        resolution = int(np.sqrt(target_v) * 1.5)
        resolution = max(32, min(resolution, 256))  # Clamp between 32 and 256
        
        print(f"\nCreating mesh with ~{target_v} vertices (resolution={resolution})...")
        
        # Voxelize mesh
        print("  Voxelizing...")
        sdf, (min_bound, max_bound) = mesh_to_voxels_fast(vertices, faces, resolution)
        
        # Apply marching cubes
        print("  Running marching cubes...")
        try:
            verts_mc, faces_mc, normals, values = marching_cubes(
                sdf, 
                level=0,
                spacing=((max_bound - min_bound) / resolution)
            )
            
            # Offset vertices to world coordinates
            verts_mc += min_bound
            
            # Save mesh
            output_file = f"{output_prefix}_{len(verts_mc)}v.obj"
            output_mesh = meshio.Mesh(
                points=verts_mc,
                cells=[("triangle", faces_mc)]
            )
            meshio.write(output_file, output_mesh)
            
            print(f"  Saved: {output_file} ({len(verts_mc)} vertices, {len(faces_mc)} faces)")
            results.append((output_file, len(verts_mc)))
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results

# Example usage
if __name__ == "__main__":
    input_mesh = "input.obj"  # Replace with your mesh file
    output_prefix = "coarse"
    
    results = coarsen_mesh_marching_cubes(
        input_mesh,
        output_prefix,
        target_vertices=[256, 512, 1024]
    )
    
    print("\n" + "="*50)
    print("Summary:")
    for filename, vertex_count in results:
        print(f"  {filename}: {vertex_count} vertices")