import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--gen_dir', type=str)
parser.add_argument('-g', '--gpu_id', type=int, default=None)
parser.add_argument('-bl', '--blender_path', type=str, default="blender")
args = parser.parse_args()

obj_paths = []
# find all obj files in the gen_dir and subdirs
if args.gen_dir is not None and os.path.exists(args.gen_dir):
    obj_paths += glob.glob(os.path.join(args.gen_dir, '**/object.obj'), recursive=True)

if len(obj_paths) == 0:
    print('No obj files found!')
    exit(1)

for i, path in enumerate(obj_paths):
    path = os.path.abspath(path)
    out_dir = os.path.join(os.path.dirname(path), f"renderings")
    print(f"Rendering {path} to {out_dir} ...")
    cmd = f"{args.blender_path} -b -P blender_render_multiview.py -- -s {path} -o {out_dir} -g {args.gpu_id}"
    os.system(cmd)
