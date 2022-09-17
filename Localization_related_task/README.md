# Object folder 2.0 envrionment for Pybullet -- Localization on the object along the normal vector (with/ without arm)

## Robot
Use panda as the platform to conduct grasp tasks.

## How to run it

1. python localize.py --num N --start M

In this file, panda robot points to certion vertex on the object along the normal vector.
Args:
--num: the serial number of object
--start: the serial number of data in the npy files

2. python localize_new.py --num N

Directly generate images for certain vertex along normal vector

Args:
--num: the serial number of object

## Object
All objects are in ./ObjectFolder/. For each object, need to define its .urdf file. 

