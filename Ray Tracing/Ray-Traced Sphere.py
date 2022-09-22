import numpy as np
import matplotlib.pyplot as plt


def norm(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

# function inputs: sphere center, sphere radius, ray origin point, ray direction vector
# function output: minimum distance if an intersection between ray and sphere, else function returns "None"

def sphere_intersect(c, r, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - c)
    c = np.linalg.norm(ray_origin - c) ** 2 - r ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

# function inputs: number of objects in scene, ray origin point, ray direction vector
# function outputs: nearest object and minimum distance at which the nearest object intersects with the ray

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = []
    for obj in objects:
        distances.append(sphere_intersect(obj['c'], obj['r'], ray_origin, ray_direction))
        nearest_object = None
        min_distance = np.inf
        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = objects[index]
        return nearest_object, min_distance

# number of pixels the screen will be split into
pixel_array = 300

max_depth = 1

# camera perspective
camera = np.array([0, 0, 1])
screen = (-1, 1, 1, -1) # left, top, right, bottom

# radar simulator
light = {'position': np.array([0, 0, 1]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [{ 'c': np.array([0, 0, -1]), 'r': 1, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.525, 0.525, 0.525]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }]

image = np.zeros((pixel_array, pixel_array, 3))
rays = []
intersection_distance_array = []
intersection_to_light_array = []
camera_to_light_array = []
power = []

# parsing through pixels from top to bottom
for i, y in enumerate(np.linspace(screen[1], screen[3], pixel_array)):
    # parsing through pixels from left to right
    for j, x in enumerate(np.linspace(screen[0], screen[2], pixel_array)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera  # receiver
        rays.append(np.linalg.norm(pixel - origin))
        direction = norm(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection_distance_array.append(min_distance)
            intersection = origin + min_distance * direction
            normal_to_surface = norm(intersection - nearest_object['c'])
            shifted_point = intersection + 1e-6 * normal_to_surface
            intersection_to_light = norm(light['position'] - shifted_point)
            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)

            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            intersection_to_light_array.append(intersection_to_light_distance)

            camera_to_light_array.append(np.linalg.norm(intersection) + intersection_to_light_distance)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                break

            illumination = np.zeros((3))

            # ambient
            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

            # specular
            intersection_to_camera = norm(camera - intersection)
            H = norm(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

            # reflection
            color += reflection * illumination
            power.append(np.sum(color) / 3)
            reflection = nearest_object['reflection']

            # new ray origin and direction
            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
    # print("%d/%d" % (i + 1, pixel_array))

plt.imsave('sphere.png', image)

plt.plot(intersection_distance_array, power)
plt.show()

f = 1*(10**9)  # 60 GHz
c = 3*(10**8)  # speed of light
phi_array = []

for i in range(0, len(intersection_distance_array)):
    d = intersection_distance_array[i]
    phi_array.append(np.mod((d / (c / f)) * 2 * np.pi, 2 * np.pi))

plt.scatter(intersection_distance_array, phi_array)
plt.show()
