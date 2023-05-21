import os
from utils.parser_util import sample_args, encoding_feat_path, encoding_log_dir, diffusion_model_path
from utils import dist_util


def sample_diffusion(args):
    from utils.triplane_util import load_triplane_data, decompose_featmaps, save_triplane_data
    from diffusion.script_util import create_model_and_diffusion_from_args
    
    # dist_util.setup_dist(args.gpu_id)

    src_data, sizes = load_triplane_data(encoding_feat_path(args.tag), device=dist_util.dev())

    model, diffusion = create_model_and_diffusion_from_args(args)
    model_path = diffusion_model_path(args.tag, args.ema_rate, args.diff_n_iters)
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev()).eval()

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    result_dir = os.path.join(args.tag, args.output)
    os.makedirs(result_dir, exist_ok=True)

    C = src_data.shape[0]
    H, W, D = sizes
    batch_size = args.diff_batch_size
    H, W, D = int(H * args.resize[0]), int(W * args.resize[1]), int(D * args.resize[2])
    print("H, W, D:", H, W, D)

    result_paths = []
    for i in range(0, args.n_samples, batch_size):
        bs = min(batch_size, args.n_samples - i)
        out_shape = [bs, C, H + D, W + D]

        cond = {'H': H, 'W': W, 'D': D}
        samples = sample_fn(model, out_shape, progress=True, model_kwargs=cond)
        samples_xy, samples_xz, samples_yz = decompose_featmaps(samples, (H, W, D))
        samples_xy = samples_xy.detach().cpu().numpy()
        samples_xz = samples_xz.detach().cpu().numpy()
        samples_yz = samples_yz.detach().cpu().numpy()

        for j in range(bs):
            save_path = os.path.join(result_dir, f"{i+j:03d}", "feat.npz")
            save_triplane_data(save_path, samples_xy[j], samples_xz[j], samples_yz[j])
            result_paths.append(save_path)
    return result_paths


def decode(args, paths):
    from encoding.model import ShapeAutoEncoder
    from utils.triplane_util import load_triplane_data
    import glob

    # dist_util.setup_dist(args.gpu_id)
    
    log_dir = encoding_log_dir(args.tag)
    ae_model = ShapeAutoEncoder(log_dir, args)
    ae_model.load_ckpt("final")

    for path in paths:
        feat_maps = load_triplane_data(path, device=dist_util.dev(), compose=False)
        feat_maps = [fm.unsqueeze(0) for fm in feat_maps]

        save_dir = os.path.dirname(path)
        if args.vox:
            ae_model.decode_voxel(save_dir, feat_maps, args.reso)
        else:
            if args.copy_mtl:
                mtl_path = glob.glob(os.path.join(os.path.dirname(args.data_path), "mesh/*.mtl"))[0]
            else:
                mtl_path = None
            ae_model.decode_texmesh(save_dir, feat_maps, args.reso, n_faces=args.n_faces, texture_reso=args.texreso,
                                    save_highres_mesh=False, n_surf_pc=-1, mtl_path=mtl_path)


if __name__ == "__main__":
    args = sample_args()
    dist_util.setup_dist(args.gpu_id)

    result_paths = sample_diffusion(args)
    decode(args, result_paths)
