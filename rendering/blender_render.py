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
parser.add_argument('-o', '--output_path', type=str, default=None, help="output path")
parser.add_argument('-az', '--azimuth', type=float, default=45, help="azimuth angle")
parser.add_argument('-el', '--elevation', type=float, default=0, help="elevation angle")
parser.add_argument('--scale', type=float, default=1.0, help="mesh scale")
parser.add_argument('--rot', type=float, default=0, help="horizontal rotation")
parser.add_argument('--light_intensity', type=float, default=2, help="")
parser.add_argument('--light_angle', type=float, default=45, help="")
parser.add_argument('--light_height', type=float, default=2, help="")
parser.add_argument('--image_resolution', nargs=2, type=int, default=(512, 512), help="resolution of image plane")
parser.add_argument('--number_of_samples', type=int, default=200, help="number of samples")
parser.add_argument('--shading', type=str, default='smooth', choices=['smooth', 'flat'])
parser.add_argument('--subdivision_iteration', type=int, default=0)
parser.add_argument('--mesh_color', type=str, default='grey', choices=['red', 'blue', 'green', 'grey'])
parser.add_argument('-g', '--gpu_id', type=int, default=0, help="gpu id")
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)
arguments = args.__dict__
if arguments['output_path'] is None:
    arguments['output_path'] = os.path.splitext(arguments['mesh_path'])[0] + '.png'

if arguments["gpu_id"] is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments["gpu_id"])

enable_cuda_devices()

outputPath = arguments['output_path'] # make it abs path for windows

## initialize blender
imgRes_x = arguments['image_resolution'][0] # recommend > 1080 (UI: Scene > Output > Resolution X)
imgRes_y = arguments['image_resolution'][1] # recommend > 1080 
numSamples = arguments['number_of_samples'] # recommend > 200 for paper images
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
# bpy.data.scenes[0].view_layers[0]['cycles']['use_denoising'] = 1

## read mesh (choose either readPLY or readOBJ)
meshPath = arguments['mesh_path']

location = [0, 0, 0]
rotation = [90, 0, arguments['rot']]
scale = [1, 1, 1]
mesh = bt.readMesh(meshPath, location, rotation, scale)
# normalize
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
mesh.location = Vector((0, 0, 0))
bbox_size = max(mesh.dimensions.x, mesh.dimensions.y, mesh.dimensions.z) / 2 * 1.03
mesh.scale = [arguments['scale'] / bbox_size] * 3
bpy.context.view_layer.update()

mesh_minz = np.array(mesh.bound_box).min(axis=0)[1] * mesh.scale[1]

# set shading (uncomment one of them)
if arguments['shading'] == 'smooth':
    bpy.ops.object.shade_smooth() # Option1: Gouraud shading
elif arguments['shading'] == 'flat':
    bpy.ops.object.shade_flat() # Option2: Flat shading
else:
    raise NotImplementedError
# bt.edgeNormals(mesh, angle = 10) # Option3: Edge normal shading

## subdivision
if arguments['subdivision_iteration'] > 0:
    bt.subdivision(mesh, level = arguments['subdivision_iteration'])

###########################################
## Set your material here (see other demo scripts)

## End material
###########################################

## set invisible plane (shadow catcher)
# bt.invisibleGround(shadowBrightness=arguments['ground_shadowBrightness'])
ground_location = [0, 0, mesh_minz - 0.01]
bt.invisibleGround(location=ground_location, shadowBrightness=0.9)


## set light
## Option1: Three Point Light System 
# bt.setLight_threePoints(radius=arguments["light_radius"], height=arguments["light_height"], 
#                         intensity=arguments["light_intensity"], softness=6, keyLoc='left')
## Option2: simple sun light
# lightAngle = arguments['light_angle'] # UI: click Sun > Transform > Rotation
def setLight_sun(location, rotation_euler, strength, shadow_soft_size = 0.05):
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x,y,z)
    if location is not None:
        bpy.ops.object.light_add(type = 'SUN', rotation = angle, location = location)
    else:
        bpy.ops.object.light_add(type = 'SUN', rotation = angle)
    lamp = bpy.data.lights['Sun']
    lamp.use_nodes = True
    # lamp.shadow_soft_size = shadow_soft_size # this is for older blender 2.8
    lamp.angle = shadow_soft_size

    lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
    return lamp

lightLocation = [2, 0, arguments["light_height"]]
lightAngle = [0, arguments["light_angle"], 0]
strength = arguments["light_intensity"]
shadowSoftness = 0.3
sun = setLight_sun(lightLocation, lightAngle, strength, shadowSoftness)

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
phi = arguments["azimuth"] / 180 * math.pi
theta = arguments["elevation"] / 180 * math.pi
camLocation = [d * math.sin(theta) * math.cos(phi), d * math.sin(theta) * math.sin(phi), d * math.cos(theta)]
lookAtLocation = (0, 0, 0)
focalLength = 45 # (UI: click camera > Object Data > Focal Length)

cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

## save rendering
print('before renderImage')
if not outputPath.endswith('.png'):
    outputPath += '.png'
bt.renderImage(outputPath, cam)

# render only geometry
color_dict = {
    # "blue": [152, 199, 255, 255],
    "blue": [144, 210, 236, 255],
    # "green": [186, 221, 173, 255],
    "green": [165, 221, 144, 255],
    # "red": [255, 189, 189, 255],
    "red": [255, 154, 156, 255],
    "grey": [134, 134, 134, 255]
}
RGBA = [x / 255.0 for x in color_dict[arguments['mesh_color']]]
# meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
# bt.setMat_plastic(mesh, meshColor)
meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
AOStrength = 0.0
mesh.data.materials.clear()
bt.setMat_balloon(mesh, meshColor, AOStrength)

outputPath_geo = outputPath.replace(".png", "_geo.png")
bt.renderImage(outputPath_geo, cam)

## save blender file so that you can adjust parameters in the UI
print('before save .blend')
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
