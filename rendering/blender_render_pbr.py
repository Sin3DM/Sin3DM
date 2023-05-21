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
parser.add_argument('--light_intensity', type=float, default=500, help="")
parser.add_argument('--light_radius', type=float, default=2, help="")
parser.add_argument('--light_height', type=float, default=10, help="")
parser.add_argument('--image_resolution', nargs=2, type=int, default=(512, 512), help="resolution of image plane")
parser.add_argument('--number_of_samples', type=int, default=200, help="number of samples")
parser.add_argument('--shading', type=str, default='smooth', choices=['smooth', 'flat'])
parser.add_argument('--subdivision_iteration', type=int, default=0)
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
def readOBJ(filePath, location, rotation_euler, scale):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)

	prev = []
	for ii in range(len(list(bpy.data.objects))):
		prev.append(bpy.data.objects[ii].name)
	bpy.ops.wm.obj_import(filepath=filePath, directory=os.path.dirname(meshPath), use_split_objects=False) # this import support PBR extension
	after = []
	for ii in range(len(list(bpy.data.objects))):
		after.append(bpy.data.objects[ii].name)
	name = list(set(after) - set(prev))[0]
	mesh = bpy.data.objects[name]

	mesh.location = location
	mesh.rotation_euler = angle
	mesh.scale = scale
	bpy.context.view_layer.update()

	return mesh 

mesh = readOBJ(meshPath, location, rotation, scale)
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
# FIXME: this is a hack to find albedo, metallic, roughness, normal images
all_img_names = list(bpy.data.images.keys())
albedo_name = None
metallic_name = None
roughness_name = None
normal_name = None
for name in all_img_names:
    if "albedo" in name:
        albedo_name = name
    if "metallic" in name:
        metallic_name = name
        bpy.data.images[name].colorspace_settings.name = 'Non-Color'
    if "roughness" in name:
        roughness_name = name
        bpy.data.images[name].colorspace_settings.name = 'Non-Color'
    if "normal" in name:
        normal_name = name
        bpy.data.images[name].colorspace_settings.name = 'Non-Color'

## End material
###########################################

## set invisible plane (shadow catcher)
# bt.invisibleGround(shadowBrightness=arguments['ground_shadowBrightness'])
ground_location = [0, 0, mesh_minz - 0.01]
bt.invisibleGround(location=ground_location, shadowBrightness=0.9)


## set light

# Three Area Light System
def setLight_threeArea(
    radius = 4,
    height = 10,
    intensity = 100,
    softness = 1):
    bpy.ops.object.light_add(type='AREA', radius=softness, location=(radius,0,height))
    KeyL = bpy.data.lights['Area']
    KeyL.energy = intensity
    bpy.ops.object.light_add(type='AREA', radius=softness, location=(0,radius,0.6*height))
    FillL = bpy.data.lights['Area.001']
    FillL.energy = intensity * 0.5
    bpy.ops.object.light_add(type='AREA', radius=softness, location=(0,-radius,height))
    RimL = bpy.data.lights['Area.002']
    RimL.energy = intensity * 0.1
    return [KeyL, FillL, RimL]
setLight_threeArea(radius=arguments["light_radius"], height=arguments["light_height"],
                   intensity=arguments["light_intensity"], softness=1.0)


## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold (optional but recommended)
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

import bpy
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

meshColor = bt.colorObj((0, 0, 0, 1), 0.5, 1.0, 1.0, 0.0, 0.0)
if albedo_name is not None:
    bt.setMat_texture(mesh, bpy.data.images[albedo_name].filepath, meshColor)
    outputPath_albedo = outputPath.replace(".png", "_albedo.png")
    bt.renderImage(outputPath_albedo, cam)

if metallic_name is not None:
    bt.setMat_texture(mesh, bpy.data.images[metallic_name].filepath, meshColor)
    outputPath_meta = outputPath.replace(".png", "_metallic.png")
    bt.renderImage(outputPath_meta, cam)

if roughness_name is not None:
    bt.setMat_texture(mesh, bpy.data.images[roughness_name].filepath, meshColor)
    outputPath_rough = outputPath.replace(".png", "_roughness.png")
    bt.renderImage(outputPath_rough, cam)

if normal_name is not None:
    bt.setMat_texture(mesh, bpy.data.images[normal_name].filepath, meshColor)
    outputPath_normal = outputPath.replace(".png", "_normal.png")
    bt.renderImage(outputPath_normal, cam)

# grey color
RGBA = [x / 255.0 for x in [134, 134, 134, 255]]
meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
AOStrength = 0.0
bt.setMat_balloon(mesh, meshColor, AOStrength)
outputPath_geo = outputPath.replace(".png", "_geo.png")
bt.renderImage(outputPath_geo, cam)


## save blender file so that you can adjust parameters in the UI
print('before save .blend')
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
