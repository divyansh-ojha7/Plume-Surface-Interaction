import numpy as np
import matplotlib.pyplot as plt
import trimesh

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
        shortlist_active_locations, index_ray, shortlist_active_index_tri =active_mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)

        for k, index in enumerate(index_ray):
            # print(active_locations[index])
            # print(shortlist_active_locations[k])
            active_locations[index] = shortlist_active_locations[k]
            active_normals[index] = active_mesh.face_normals[shortlist_active_index_tri[k]]

        mask = np.abs(active_locations - ray_origins) < np.abs(locations - ray_origins)
        # print(mask)
        locations[mask] = active_locations[mask]
        normals[mask] = active_normals[mask]
        # print(normals)
        # print(active_normals)
        intersected_mesh[np.transpose(mask[:, 0])] = i

    return locations, normals, intersected_mesh

# number of pixels the screen will be split into
width = 350
height = 350

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

mesh_list = [trimesh.load('cylinder_4.stl')]

# number of reflections allowed
max_depth = 3

# radar simulator
# light = { 'position': np.array([0, 0, 1]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) } # color vectors can be replaced by scalar values: DONE
light = {'position': np.array([0, 0, 1]), 'ambient': 1, 'diffuse': 1, 'specular': 1}  # color

objects = {'file': 'cylinder_4.stl', 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.5, 0.5, 0.5]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
objects = {'mesh_index': 0, 'ambient': 0.1, 'diffuse': 0.5, 'specular': 1, 'shininess': 100, 'reflection': 0.5 }

image3 = np.zeros((width, height, 3))

intersection_distance_array = []
intersection_to_light_array = []
camera_to_light_array = []
distance = []
power = []

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

# parsing through pixels from top to bottom
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    # parsing through pixels from left to right
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            intersection = locations[i * width + j]
            distance.append(np.linalg.norm(intersection - ray_origins[0]))
            normal = normals[i * width + j]

            # intersection_distance_array.append(distance)
            # normal_to_surface = norm(intersection − normal)
            intersection_to_light = normalize(light['position'] - intersection)  # change the name of "norm" function: DON

            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            # intersection_to_light_array.append(intersection_to_light_distance)

            # camera_to_light_array.append(np.linalg.norm(intersection) + intersection_to_light_distance)

            illumination = np.zeros((3))

            # ambient
            illumination += objects.get('ambient') * light['ambient']

            # diffuse
            illumination += objects.get('diffuse') * light['diffuse'] * np.dot(intersection_to_light, normal)

            # specular
            intersection_to_camera = -1 * ray_directions[i * width + j]  # replaced this with ray_directions[i * width + j]: DONE
            H = normalize(intersection_to_light + intersection_to_camera)  # H is halfway vector in the Blinn−Phong reflection model: DONE
            illumination += objects.get('specular') * light['specular'] * np.dot(normal, H) ** (objects.get('shininess') / 4)

            # reflection
            color += reflection * illumination
            power.append(np.sum(color) / 3)
            reflection *= objects.get('reflection')

        image3[i, j] = np.clip(color, 0, 1)
    # print("%d/%d" % (i + 1, pixel_array))

plt.imsave('triangle_mesh.png', image3)

plt.scatter(distance, power)
plt.show()
