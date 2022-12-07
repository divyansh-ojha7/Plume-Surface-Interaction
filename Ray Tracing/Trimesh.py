# import numpy as np
# import trimesh
#
# # load a file by name or from a buffer
# mesh = trimesh.load_mesh('cylinder_4.stl')
#
# # is the current mesh watertight?
# print(mesh.is_watertight)
#
# # what's the euler number for the mesh?
# mesh.euler_number
#
#
# # since the mesh is watertight, it means there is a
# # volumetric center of mass which we can set as the origin for our mesh
# mesh.vertices -= mesh.center_mass
#
# # what's the moment of inertia for the mesh?
# print(mesh.moment_inertia)
import math

import trimesh

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y


# Scene
mesh = trimesh.load_mesh('Cube.stl')

width = 5
height = 5

camera = np.array([0, 0, 20])
ratio = float(width) / height
screen = (-10, 10 / ratio, 10, - 10/ ratio)  # left, top, right, bottom
light = np.array([0, 10, 20])



# mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
matrix = [[1,0,0,0],
        [0,1,0,0],
        [0,0,1,-10],
        [0,0,0,0]]
mesh.apply_transform(trimesh.transformations.transform_around(matrix, [0,0,0]))

mesh.bounding_box.extents

fig = plt.figure()

ax = mplot3d.Axes3D(fig)

# Object
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.triangles,facecolor=np.array([(1., 1., 1., 1)]),edgecolor=np.array([(0., 0., 0., 1)])))

# Add Camera
ax.scatter(camera[0], camera[1], camera[2])
ax.text(camera[0], camera[1], camera[2], '%s' % ("Camera"), size=10, zorder=1, color='k')

# Add Light
ax.scatter(light[0], light[1], light[2])
ax.text(light[0], light[1], light[2], '%s' % ("Light"), size=10, zorder=1, color='k')

# Add Screen
screen_x = [screen[0], screen[0], screen[2], screen[2]]
screen_y = [screen[1], screen[3], screen[3], screen[1]]
screen_z = [0, 0, 0, 0]
vertices = [list(zip(screen_x, screen_y, screen_z))]
poly = Poly3DCollection(vertices, alpha=0.8)
ax.add_collection3d(poly)
ax.text(screen[0], screen[1], 0, '%s' % ("Screen"), size=10, zorder=1, color='k')

# Grid On Screen
point_data = []
for yy in np.linspace(screen[1], screen[3], height):
    point_col = []
    for xx in np.linspace(screen[0], screen[2], width):
        point_col.append(Point(xx, yy))
    point_data.append(point_col)


VecEndCamera_x = []
VecEndCamera_y = []
for x in range(0, len(point_data)-1, 1):
    for y in range(0, len(point_data[0])-1, 1):
        screen_x = [point_data[x][y].x, point_data[x+1][y].x, point_data[x+1][y+1].x, point_data[x][y+1].x]
        screen_y = [point_data[x][y].y, point_data[x+1][y].y, point_data[x+1][y+1].y, point_data[x][y+1].y]
        vertices = [list(zip(screen_x, screen_y, screen_z))]
        poly = Poly3DCollection(vertices, alpha=0.8)
        scalar = np.random.rand(3, 4)
        averaged_scalar = np.mean(scalar, axis=1)
        m = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
        m.set_array(scalar)
        m.set_clim(vmin=np.amin(averaged_scalar), vmax=np.amax(averaged_scalar))
        rgba_array = m.to_rgba(averaged_scalar)
        poly.set_facecolor([(i[0], i[1], i[2]) for i in rgba_array])
        ax.add_collection3d(poly)

        VecEndCamera_x.append((point_data[x][y].x + point_data[x+1][y+1].x) / 2)
        VecEndCamera_y.append((point_data[x][y].y + point_data[x+1][y+1].y) / 2)


for i in range(len(VecEndCamera_x)):
    ax.plot([camera[0], VecEndCamera_x[i]], [camera[1],VecEndCamera_y[i]],zs=[camera[2],0])
    ax.plot([light[0], VecEndCamera_x[i]], [light[1], VecEndCamera_y[i]], zs=[light[2], 0])



# screen_x = [-20, -10, 10, 10]
# screen_y = [-20, 10, 10, -10]
# vertices = [list(zip(screen_x, screen_y, screen_z))]
# poly = Poly3DCollection(vertices, alpha=0.8)
# ax.add_collection3d(poly)

# XYZ Properties
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25)

plt.show()
