import math

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import seaborn as sns


def normalize(vector):
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def triangle_intersect(mesh_list, ray_origins, ray_directions):
    locations = np.full((len(ray_origins), 3), np.inf)
    normals = np.full((len(ray_origins), 3), 0.0)
    intersected_mesh = np.full((len(ray_origins), 1), np.nan)

    for i in range(0, len(mesh_list)):
        active_mesh = mesh_list[i]
        active_locations = np.full((len(ray_origins), 3), np.inf)
        active_normals = np.full((len(ray_origins), 3), 0.0)
        active_index_tri = np.full((len(ray_origins), 1), np.nan)
        shortlist_active_locations, index_ray, shortlist_active_index_tri = active_mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)

        for k, index in enumerate(index_ray):
            active_locations[index] = shortlist_active_locations[k]
            active_normals[index] = active_mesh.face_normals[shortlist_active_index_tri[k]]

        mask = np.abs(active_locations - ray_origins) < np.abs(locations - ray_origins)
        locations[mask] = active_locations[mask]
        normals[mask] = active_normals[mask]
        intersected_mesh[np.transpose(mask[:, 0])] = i

    return locations, normals, intersected_mesh

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [triangle_intersect(mesh_list, ray_origins, ray_directions) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def PowerVsDistance(power_type):
    import matplotlib.pyplot as plt

    # Creating distribution
    i = 0
    new_dis = []
    new_pow = []

    for x in distance:
        if x != math.inf:
            new_dis.append(distance[i])
            if power_type == "Power":
                new_pow.append(power[i])
            if power_type == "Phasor":
                new_pow.append(phasor_angle[i])
        i += 1
    x = new_dis
    y_data = [0] * 20
    bins = np.linspace(min(x), max(x), 20)

    for i in x:
        count = 0
        for j in bins:
            if(i < j):
                y_data[count] += new_pow[count]
                break
            count += 1

    round_to_hundrends = [round(num, 1) for num in bins]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=round_to_hundrends,y=y_data,ax=ax, palette = 'plasma')
    ax.tick_params(labelsize=8, length=0)
    plt.box(False)  # removing border lines


    # # Creating histogram
    # fig, axs = plt.subplots(1, 1,
    #                         figsize=(10, 7),
    #                         tight_layout=True)
    #
    # # Remove axes splines
    # for s in ['top', 'bottom', 'left', 'right']:
    #     axs.spines[s].set_visible(False)
    #
    # # Remove x, y ticks
    # axs.xaxis.set_ticks_position('none')
    # axs.yaxis.set_ticks_position('none')
    #
    # # Add padding between axes and labels
    # axs.xaxis.set_tick_params(pad=5)
    # axs.yaxis.set_tick_params(pad=10)
    #
    # # Add x, y gridlines
    # axs.grid(b=True, color='grey',
    #          linestyle='-.', linewidth=0.5,
    #          alpha=0.6)
    #
    # # Creating histogram
    # tests = [10,6,10,2]
    # N, bins, patches = axs.hist(x, bins=20)
    # # plt.ylim((None, 100))
    #
    # # Setting color
    # fracs = ((N ** (1 / 5)) / N.max())
    # norm = colors.Normalize(fracs.min(), fracs.max())
    #
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)

    # Adding extra features
    plt.xlabel("Distance")
    if power_type == "Power":
        plt.ylabel("Power")
        plt.title('Power vs Distance')
    if power_type == "Phasor":
        plt.ylabel("Phasor")
        plt.title('Phasor vs Distance')

    # Show plot
    plt.show()

def VirtualEnvironment():
    import trimesh
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Scene
    mesh = trimesh.load_mesh('cylinder_4.stl')
    mesh2 = trimesh.load_mesh('Cube.STL')

    light = np.array([0, 0, 20])

    # mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    matrix = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, -10],
              [0, 0, 0, 1]]
    mesh.apply_transform(trimesh.transformations.transform_around(matrix, [0, 0, 0]))
    matrix = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, -5],
              [0, 0, 0, 0]]
    mesh2.apply_transform(trimesh.transformations.transform_around(matrix, [0, 0, 0]))

    mesh.bounding_box.extents
    mesh2.bounding_box.extents


    fig = plt.figure()

    ax = mplot3d.Axes3D(fig)

    # Object
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.triangles, facecolor=np.array([(1., 1., 1., 1)]),
                                                       edgecolor=np.array([(0., 0., 0., 1)])))
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh2.triangles, facecolor=np.array([(1., 1., 1., 1)]),
                                                       edgecolor=np.array([(0., 0., 0., 1)])))

    # Add Camera
    ax.scatter(camera[0], camera[1], camera[2])
    ax.text(camera[0], camera[1], camera[2], '%s' % ("Camera"), size=10, zorder=1, color='k')

    # Add Light
    ax.scatter(light[0], light[1], light[2])
    ax.text(light[0], light[1], light[2], '%s' % ("Light"), size=10, zorder=1, color='k')

    # Add Screen
    screen_z = [0, 0, 0, 0]
    ax.text(screen[0], screen[1], 0, '%s' % ("Screen"), size=10, zorder=1, color='k')

    gridStepSizeX = (-screen[0] / width) + ((-screen[0] / width) / 4)
    gridStepSizeY = (screen[2] / height) + ((screen[2] / height) / 4)
    locationIterator = 0
    image_x = 0
    image_y = 0
    for yy in np.linspace(screen[1], screen[3], height):
        for xx in np.linspace(screen[0], screen[2], width):
            screen_x = [xx - gridStepSizeX, xx - gridStepSizeX, xx + gridStepSizeX, xx + gridStepSizeX]
            screen_y = [yy - gridStepSizeY, yy + gridStepSizeY, yy + gridStepSizeX, yy - gridStepSizeY]
            vertices = [list(zip(screen_x, screen_y, screen_z))]
            poly = Poly3DCollection(vertices, alpha=0.8)
            if (math.isnan(image_power[image_y][image_x][0])):
                poly.set_facecolor([(0, 0, 0)])
            else:
                poly.set_facecolor([(image_power[image_y][image_x][0], 1, 1)])
            ax.add_collection3d(poly)

            if (locations[locationIterator][0] != math.inf):
                ax.plot([camera[0], locations[locationIterator][0]], [camera[1], locations[locationIterator][1]],
                        zs=[camera[2], locations[locationIterator][2]])

            locationIterator += 1
            image_x += 1
        image_y += 1
        image_x = 0

    # XYZ Properties
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25)

    plt.show()

wavelength = 0.005 * 10

# number of pixels the screen will be split into
width = 25
height = 25

camera = np.array([0, 0, 20])
ratio = float(width) / height
screen = (-10, 10 / ratio, 10, -10 / ratio)  # left, top, right, bottom

mesh = trimesh.load_mesh('cylinder_4.stl')
matrix = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, -10],
          [0, 0, 0, 1]]
mesh.apply_transform(trimesh.transformations.transform_around(matrix, [0, 0, 0]))

# mesh2 = trimesh.load_mesh('Cube.stl')
# matrix = [[1,0,0,0],
#         [0,1,0,0],
#         [0,0,1,-5],
#         [0,0,0,0]]
# mesh2.apply_transform(trimesh.transformations.transform_around(matrix, [0,0,0]))

mesh_list = [mesh]
# mesh_list.append(mesh2)

# number of reflections allowed
max_depth = 3

# radar simulator
# light = { 'position': np.array([0, 0, 1]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) } # color vectors can be replaced by scalar values: DONE
light = {'position': np.array([0, 0, 40]), 'ambient': 1000, 'diffuse': 1000, 'specular': 1000}  # color

objects = [{'mesh_index': 0, 'ambient': 0.1, 'diffuse': 0.5, 'specular': 1, 'shininess': 100, 'reflection': 0.5},
            {'mesh_index': 1, 'ambient': 0, 'diffuse': 0, 'specular': 0, 'shininess': 100, 'reflection': 0.5}]

image_power = np.zeros((width, height, 1))
image_phasor = np.zeros((width, height, 1))

intersection_distance_array = []
camera_to_light_array = []
distance = []
power = []
phasor_angle = []
phasor = []

ray_origin = []
ray_direction = []
for yy in np.linspace(screen[1], screen[3], height):
    for xx in np.linspace(screen[0], screen[2], width):
        # screen is on origin
        pixel = np.array([xx, yy, 0])
        origin = camera  # receiver
        ray_origin.append(origin)
        ray_direction.append(pixel - origin)

ray_origins = np.array(ray_origin)
ray_directions = np.array(ray_direction)

locations, normals, intersected_mesh = triangle_intersect(mesh_list, ray_origins, ray_directions)

phase = []
# parsing through pixels from top to bottom
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    # parsing through pixels from left to right
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        for k in range(max_depth):
            # check for intersections
            intersection = locations[i * width + j]
            local_intersected_mesh_index = intersected_mesh[i * width + j]
            if math.isnan(local_intersected_mesh_index):
                power.append(0)
                phasor_angle.append(0)
                break
            int(local_intersected_mesh_index)
            distance.append(np.linalg.norm(intersection - ray_origins[0]))
            normal = normals[i * width + j]

            intersection_to_light = normalize(light['position'] - intersection)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            intersection_to_camera_distance = np.linalg.norm(camera - intersection)
            total_distance = intersection_to_camera_distance + intersection_to_light_distance


            illumination = np.zeros(1) * np.exp(1j*2*np.pi*total_distance/wavelength)

            # ambient
            illumination += (objects[int(local_intersected_mesh_index[0])].get('ambient') * (light['ambient'] * (1/(intersection_to_light_distance**2)))) * np.exp(1j*2*np.pi*total_distance/wavelength)

            # diffuse
            illumination += objects[int(local_intersected_mesh_index[0])].get('diffuse') * (light['diffuse'] * (1/(intersection_to_light_distance**2))) * np.dot(intersection_to_light, normal) * np.exp(1j*2*np.pi*total_distance/wavelength)

            # specular
            intersection_to_camera = -1 * ray_directions[i * width + j]
            H = normalize(intersection_to_light + intersection_to_camera)  # H is halfway vector in the Blinn−Phong reflection model

            illumination += objects[int(local_intersected_mesh_index[0])].get('specular') * (light['specular'] * (1/(intersection_to_light_distance**2))) * np.dot(normal, H) ** (objects[int(local_intersected_mesh_index[0])].get('shininess') / 4) * np.exp(1j*2*np.pi*total_distance/wavelength)
            # reflection
            power.append(np.abs(illumination))
            phasor_angle.append(np.angle(illumination))
            phasor.append(illumination)

        image_power[i, j] = np.clip(power[-1], 0, 1)
        image_phasor[i, j] = np.clip(phasor_angle[-1], 0, 1)
    # print("%d/%d" % (i + 1, pixel_array))

# Save image
plt.imsave('triangle_mesh_power.png', np.dstack([image_power, image_power, image_power]))
plt.imsave('triangle_mesh_phasor.png', np.dstack([image_phasor, image_phasor, image_phasor]))

print(len(power))
print(len(distance))
# PowerVsDistance("Power")
# PowerVsDistance("Phasor")
# VirtualEnvironment()
