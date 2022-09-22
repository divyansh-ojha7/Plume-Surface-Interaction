import numpy as np
import trimesh

# load a file by name or from a buffer
mesh = trimesh.load_mesh('cylinder_4.stl')

# is the current mesh watertight?
print(mesh.is_watertight)

# what's the euler number for the mesh?
mesh.euler_number


# since the mesh is watertight, it means there is a
# volumetric center of mass which we can set as the origin for our mesh
mesh.vertices -= mesh.center_mass

# what's the moment of inertia for the mesh?
print(mesh.moment_inertia)
