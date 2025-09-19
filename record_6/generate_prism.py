import trimesh

def make_truncated_square_pyramid(base_size=1.0, top_size=0.5, height=1.0):
    """
    Generate a truncated square pyramid (frustum) mesh with square base and smaller top.

    Parameters
    ----------
    base_size : float
        Side length of the bottom square
    top_size : float
        Side length of the top square
    height : float
        Height of the prism

    Returns
    -------
    mesh : trimesh.Trimesh
        The generated mesh
    """
    # Half sizes for easier vertex definitions
    b = base_size / 2.0
    t = top_size / 2.0
    h = height

    # Define vertices: bottom square (z=0), top square (z=h)
    vertices = [
        [-b, -b, 0], [ b, -b, 0], [ b,  b, 0], [-b,  b, 0],   # bottom
        [-t, -t, h], [ t, -t, h], [ t,  t, h], [-t,  t, h]    # top
    ]

    # Define faces (triangles) using vertex indices
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # sides
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ]

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh

# Example usage
mesh = make_truncated_square_pyramid(base_size=1.0, top_size=0.75, height=1.0)

# Show interactive viewer
mesh.show()

# Export to OBJ for MuJoCo
ex_path = "sq_prism.obj"

mesh.export(ex_path)
