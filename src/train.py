from utils.parser_util import train_args, encoding_log_dir, diffusion_log_dir, encoding_feat_path
from utils import dist_util
from utils.common_util import seed_all
import os
import numpy as np


def train_ae(args):
    from encoding.model import ShapeAutoEncoder
    from utils.triplane_util import save_triplane_data

    print("[Training autoencoder]")

    # seed_all(0)
    # dist_util.setup_dist(args.gpu_id)

    assert args.data_path is not None
    log_dir = encoding_log_dir(args.tag)
    ae_model = ShapeAutoEncoder(log_dir, args)
    ae_model.train(args.data_path)

    feat_maps = ae_model.encode()
    print("feat maps shape:", [fm.shape for fm in feat_maps])
    save_path = encoding_feat_path(args.tag)
    feat_maps_np = [fm.squeeze(0).detach().cpu().numpy() for fm in feat_maps]
    save_triplane_data(save_path, feat_maps_np[0], feat_maps_np[1], feat_maps_np[2])
    
    # save mesh
    save_dir = os.path.join(log_dir, "rec")
    ae_model.decode_texmesh(save_dir, feat_maps, 256)
    

def train_diffusion(args):
    from utils.triplane_util import load_triplane_data, get_data_iterator
    from utils.common_util import seed_all
    from diffusion.script_util import create_model_and_diffusion_from_args
    from diffusion.resample import create_named_schedule_sampler
    from diffusion.train_util import TrainLoop
    from diffusion import logger

    print("[Training diffusion]")

    # seed_all(0)
    # dist_util.setup_dist(args.gpu_id)

    log_dir = diffusion_log_dir(args.tag)
    logger.configure(dir=log_dir)

    logger.log("creating data loader...")
    src_data, sizes = load_triplane_data(encoding_feat_path(args.tag), device=dist_util.dev())
    data_iter = get_data_iterator(src_data, sizes, args.diff_batch_size)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_from_args(args)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_iter,
        batch_size=args.diff_batch_size,
        microbatch=-1,
        lr=args.diff_lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=False,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.diff_n_iters,
    ).run_loop()


if __name__ == '__main__':
    args = train_args()
    
    seed_all(0)
    dist_util.setup_dist(args.gpu_id)

    if args.only_enc:
        train_ae(args)
    else:
        if args.enc_log is None:
            train_ae(args)
        train_diffusion(args)
