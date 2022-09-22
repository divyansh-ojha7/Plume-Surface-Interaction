# IMPORTS
import math
from math import sqrt, pi
import time
import PIL, PIL.Image

class Vector(object):
    """
    Vector Analysis
    Inputs:
        - X coordinate
        - Y coordinate
        - Z coordinate
    """

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def dot(self, b):
        return self.x * b.x + self.y * b.y + self.z * b.z

    def cross(self, b):
        return (self.y * b.z - self.z * b.y, self.z * b.x - self.x * b.z, self.x * b.y - self.y * b.x)

    def magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normal(self):
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag, self.z / mag)

    def __repr__(self):
        return str((self.x, self.y, self.z))

    def __add__(self, b):
        return Vector(self.x + b.x, self.y + b.y, self.z + b.z)

    def __sub__(self, b):
        return Vector(self.x - b.x, self.y - b.y, self.z - b.z)

    def __mul__(self, b):
        assert type(b) == float or type(b) == int
        return Vector(self.x * b, self.y * b, self.z * b)


class Sphere(object):
    """
    Creates a Sphere Object
    Inputs:
        - Vector for the center of the sphere
        - Radius
        - Color
        - texture
    """

    def __init__(self, center, radius, color, texture):
        self.c = center
        self.r = radius
        self.col = color
        if texture:
            self.addtexture(texture)
        else:
            self.texture = None


    def normal(self, b):
        return (b - self.c).normal()


class Ray(object):
    """
    Creates a Ray
    Inputs:
        - Starting point of the ray
        - Vector of the direction of the ray
    """
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction


class Intersection(object):
    """
    Checks Intersection and Stores the Intersection point
    Inputs:
        - Point
        - Distance
        - Noramal Vector
        - List of Objects
    """
    def __init__(self, point, distance, normal, obj):
        self.p = point
        self.d = distance
        self.n = normal
        self.obj = obj

def trace(ray, objects, light):
    pass

class Color(Vector):
    """
    Create Colors
    Inputs:
        - Red Value
        - Green Value
        - Blue Value
    """
    def __init__(self, redVal, greenVal, blueVal):
        self.red = redVal
        self.green = greenVal
        self.blue = blueVal


class LightSource(Vector):
    """
    Create a Lightsource
    Inputs:
        - Lightsource Position
        - Lightsource Angle X
        - Lightsource Angle Y
        - Lightsource Angle Z
    """

    def __init__(self, lightSourcePos, xangle, yangle, zangle):
        self.pos = lightSourcePos
        self.xangle = xangle
        self.yangle = yangle
        self.zangle = zangle



class Camera:
    """
    Create a Camera
    Inputs:
        - Camera Position
        - Camera Angle X
        - Camera Angle Y
        - Camera Angle Z
    """

    def __init__(self, cameraPos, xangle, yangle, zangle):
        self.pos = cameraPos
        self.xangle = xangle
        self.yangle = yangle
        self.zangle = zangle


def renderScene(camera, lightSource, objs, imagedims, savepath):
    """
    Create a render of a scene
    Inputs:
        - Camera
        - Lightsource
        - List of Objects
        - Image Size
        - Location to save image
    """
    imgwidth, imgheight = imagedims
    img = PIL.Image.new("RGB", imagedims)
    print("rendering 3D scene")
    t = time.clock()
    for x in range(imgwidth):
        for y in range(imgheight):
            ray = Ray(camera.pos, (Vector(x / camera.zoom + camera.xangle, y / camera.zoom + camera.yangle,
                                          0) - camera.pos).normal())
            col = trace(ray, objs, lightSource, 10)
            img.putpixel((x, imgheight - 1 - y), col)
    print("time taken", time.clock() - t)
    img.save(savepath)


# Colors
red = (255, 0, 0)
yellow = (255, 255, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
grey = (120, 120, 120)
white = (255, 255, 255)
purple = (200, 0, 200)


if __name__ == "__main__":
    """
    Main Runner
    """
    def sphereTest():
        print("Ray Tracing Sphere")

        # Create the environment
        imagedims = (500, 500)
        savepath = "testing/Sphere.png"
        objs = []
        objs.append(Sphere(Vector(-2, 0, -10), 2, Vector(*green)))
        objs.append(Sphere(Vector(2, 0, -10), 3.5, Vector(*red)))
        objs.append(Sphere(Vector(0, -4, -10), 3, Vector(*blue)))
        lightSource = LightSource(0, 10, 0)
        camera = Camera(Vector(0, 0, 20))

        # RENDER
        renderScene(camera, lightSource, objs, imagedims, savepath)