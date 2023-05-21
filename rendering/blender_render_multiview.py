import sys
import json
import numpy as np
import math
import os, bpy, bmesh
from mathutils import Vector
import argparse
BLENDERTOOLBOX_PATH = "BlenderToolbox" # change this to “your/path/to/BlenderToolbox/", https://github.com/HTDerekLiu/BlenderToolbox
sys.path.append(BLENDERTOOLBOX_PATH)
import BlenderToolBox as bt
this_dir = os.path.dirname(bpy.data.filepath)
if not this_dir in sys.path:
    sys.path.append(this_dir)
from blender_utils import enable_cuda_devices

# blender --background --python blender_render.py -- -s {path to mesh file} -c {config name}
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--mesh_path', type=str, required=True, help="path to mesh file")
parser.add_argument('-o', '--output_dir', type=str, required=True, help="output dir")
parser.add_argument('--image_resolution', nargs=2, type=int, default=(512, 512), help="resolution of image plane")
parser.add_argument('--number_of_samples', type=int, default=128, help="number of samples")
parser.add_argument('-g', '--gpu_id', type=int, default=None, help="gpu id")
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)
arguments = args.__dict__

if arguments["gpu_id"] is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments["gpu_id"])

enable_cuda_devices()

## initialize blender
imgRes_x = arguments['image_resolution'][0] # recommend > 1080 (UI: Scene > Output > Resolution X)
imgRes_y = arguments['image_resolution'][1] # recommend > 1080 
numSamples = arguments['number_of_samples'] # recommend > 200 for paper images
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
bpy.data.scenes[0].view_layers[0]['cycles']['use_denoising'] = 1

## read mesh (choose either readPLY or readOBJ)
meshPath = arguments['mesh_path']

location = [0, 0, 0]
rotation = [90, 0, 0]
scale = [1, 1, 1]
mesh = bt.readMesh(meshPath, location, rotation, scale)
# normalize
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
mesh.location = Vector((0, 0, 0))
bbox_size = max(mesh.dimensions.x, mesh.dimensions.y, mesh.dimensions.z) / 2 * 1.03
mesh.scale = [1 / bbox_size, 1 / bbox_size, 1 / bbox_size]
bpy.context.view_layer.update()


###########################################
## Set your material here (see other demo scripts)

## End material
###########################################

## set light
## Option1: Three Point Light System 
# bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')
## Option2: simple sun light
# lightAngle = arguments['light_angle'] # UI: click Sun > Transform > Rotation
# lightLocation = [2, 2, 2]
# lightAngle = [0, 0, 0]
# strength = 2
# shadowSoftness = 0.3
# sun = bt.setLight_sun(lightLocation, lightAngle, strength, shadowSoftness)

bpy.ops.object.light_add(type='AREA')
light2 = bpy.data.lights['Area']
light2.energy = 30000
bpy.data.objects['Area'].location[2] = 1.5
bpy.data.objects['Area'].scale[0] = 100
bpy.data.objects['Area'].scale[1] = 100
bpy.data.objects['Area'].scale[2] = 100

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold (optional but recommended)
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

bpy.context.scene.world.light_settings.use_ambient_occlusion = True  # turn AO on
bpy.context.scene.world.light_settings.ao_factor = 0.5  # set it to 0.5

## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
# camLocation = arguments['camLocation']
# front 45 degree
d = 3.0
view_angles = [
    (0, 45), (45, 45), (90, 45), (135, 45), (180, 45), (225, 45), (270, 45), (315, 45),
]

for i, (azimuth, elevation) in enumerate(view_angles):
    phi = azimuth / 180 * math.pi
    theta = elevation / 180 * math.pi
    camLocation = [d * math.sin(theta) * math.cos(phi), d * math.sin(theta) * math.sin(phi), d * math.cos(theta)]
    lookAtLocation = (0, 0, 0)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)

    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

    ## save rendering
    outputPath = os.path.join(arguments['output_dir'], f"{i:03d}.png")
    bt.renderImage(outputPath, cam)
