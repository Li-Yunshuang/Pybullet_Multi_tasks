# Object folder 2.0 envrionment for Pybullet -- Panda with Gel_sight

## Robot

This is a special Panda robot with a gelsight sensor on the left finger. We add gelsight to panda in ./franka_panda/panda_1.urdf 

## Status

Currently, have finished fixed the gelsight onto Panda. And we set the center of gelsight to be the endpose of the whole arm so taht we could control the position of the center of Gelsight and the rotaion of the endpose is along the center of the Gelsight. 

TODO:

Fixed the rendering of Taxim well with the gelsight on Panda.
 
## How to run it

1. Install Pybullet and the environment for Object folder 2.0.

2. run "python Gelsight.py" to start up Pybullet envrioment.

3. Then you'll see the gripper with gelsight rotating along the center of the Gelsight.

4. TODO: Rendering the tactile image with Taxim at the same time.
