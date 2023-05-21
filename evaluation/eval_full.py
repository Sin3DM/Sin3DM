import os
import random
import glob
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from patch_utils import eval_LP_given_paths, eval_Div_given_paths
from ssfid import eval_SSFID_given_paths
from sifid import calculate_multiview_sifid_given_paths
from lpips import calculate_multiview_lpips_given_paths


if __name__ == "__main__":
    # reference data:
    #   - *.npz: SDF grid
    #   - renderings: rendered images
    #     - xxx.png: at i-th view
    #   - *.obj: mesh (optional)
    # generated data:
    #   samples_i_folder:
    #     - *voxel.npz: voxel grid at resolution 128
    #     - renderings: rendered images
    #       - xxx.png: at i-th view
    #     - object.obj, object.mtl, object.png: textured mesh
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help='generated data folder')
    parser.add_argument('-r', '--ref', type=str, required=True, help='reference data folder')
    parser.add_argument('--patch_size', type=int, default=11, help='patch size')
    parser.add_argument('--stride', type=int, default=5, help='patch stride. By default, half of patch size.')
    parser.add_argument('--patch_num', type=int, default=1000, help='max number of patches sampled from generated shapes.')
    parser.add_argument('-o', '--output', type=str, default=None, help='result save path')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, help="which gpu to use. -1 for CPU.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device(f"cuda:{args.gpu_ids}" if args.gpu_ids >= 0 else "cpu")
    args.stride = args.patch_size // 2 if args.stride is None else args.stride

    random.seed(1234)
    result_dict = {}

    ref_sdf_path = glob.glob(os.path.join(args.ref, '*.npz'))[0]

    # geometry evaluation
    gen_vox_paths = glob.glob(os.path.join(args.src, '*/*voxel.npz'))
    gen_vox_paths = sorted(gen_vox_paths)

    # SSFID
    ssfid_dict = eval_SSFID_given_paths(gen_vox_paths, ref_sdf_path, device=device)
    print(ssfid_dict)
    result_dict.update(ssfid_dict)

    # LP
    lp_dict = eval_LP_given_paths(gen_vox_paths, ref_sdf_path, args.patch_size, args.stride, args.patch_num, device=device)
    print(lp_dict)
    result_dict.update(lp_dict)

    # Div
    div_dict = eval_Div_given_paths(gen_vox_paths, device=device)
    print(div_dict)
    result_dict.update(div_dict)

    # renderings evaluation
    ref_render_dir = os.path.join(args.ref, 'renderings')
    n_views = len(os.listdir(ref_render_dir))

    gen_render_dirs = glob.glob(os.path.join(args.src, '*/renderings'))
    gen_render_dirs = sorted(gen_render_dirs)

    sifid_dict64 = calculate_multiview_sifid_given_paths(gen_render_dirs, ref_render_dir, device=device, dims=64)
    print(sifid_dict64)
    result_dict.update(sifid_dict64)

    sifid_dict192 = calculate_multiview_sifid_given_paths(gen_render_dirs, ref_render_dir, device=device, dims=192)
    print(sifid_dict192)
    result_dict.update(sifid_dict192)

    lpips_dict = calculate_multiview_lpips_given_paths(gen_render_dirs, device=device)
    print(lpips_dict)
    result_dict.update(lpips_dict)

    # save results
    save_path = args.output if args.output is not None else args.src + f'_eval.json'
    with open(save_path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)
