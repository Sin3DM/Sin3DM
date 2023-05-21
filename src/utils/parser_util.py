import argparse
import os
import json


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument("--tag", type=str, required=True, help="checkpoint directory")
    group.add_argument("-g", "--gpu_id", default=0, type=int, help="Device id to use.")
    group.add_argument('--only_enc', action='store_true', help="")


def add_encoding_training_options(parser):
    group = parser.add_argument_group("encoding")
    group.add_argument("--data_path", type=str, help="path to source data")
    group.add_argument("--enc_batch_size", type=int, default=65536, help="batch size")
    group.add_argument("--fm_reso", type=int, default=128, help="feature map resolution")
    group.add_argument("--sdf_renorm", type=int, default=0, help="renormalize sdf values to [-1, 1]")
    group.add_argument("--data_type", type=str, default="sdftex", choices=["sdf", "sdftex", "sdfpbr"], help="data type")
    
    group.add_argument("--enc_net_type", type=str, default="skip", help="network type")
    group.add_argument("-fdg", "--fdim_geo", type=int, default=4, help="geometry feature dimension")
    group.add_argument("-fdt", "--fdim_tex", type=int, default=8, help="texture feature dimension")
    group.add_argument("-fdup", "--fdim_up", type=int, default=64, help="conv feature dimension")
    group.add_argument("-hd", "--hidden_dim", type=int, default=256, help="mlp hidden dimension")
    group.add_argument("-nh", "--n_hidden_layers", type=int, default=4, help="mlp hidden layers")

    group.add_argument("--enc_n_iters", type=int, default=25000, help="total number of epochs to train")
    group.add_argument("--enc_lr", type=float, default=5e-3, help="initial learning rate")
    group.add_argument("--enc_lr_decay", type=float, default=0.1, help="initial learning rate")
    group.add_argument("--enc_lr_split", type=float, default=0.2, help="")
    group.add_argument("--vol_ratio", type=float, default=0.1, help="vol points ratio")
    group.add_argument("--tex_threshold_ratio", type=float, default=0.999, help="tex threshold ratio")
    group.add_argument("--tex_weight", type=float, default=1.0, help="tex weight")
    group.add_argument("--sdf_loss", type=str, default="weightedl1", choices=["l1", "weightedl1"])
    group.add_argument("--tex_loss", type=str, default="l1", choices=["l1", "l2", "huber"])


def add_diffusion_training_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--enc_log", type=str, default=None, help="path to source data")
    group.add_argument("--diff_batch_size", type=int, default=32, help="batch size for diffusion training")
    group.add_argument("--diff_net_type", type=str, default="unet_small", help="network type")
    group.add_argument("--diff_lr", type=float, default=5e-4, help="initial learning rate for diffusion training")
    group.add_argument("--diff_n_iters", type=int, default=25000, help="lr anneal steps for diffusion training")
    group.add_argument("--schedule_sampler", type=str, default="uniform", help="schedule sampler")
    group.add_argument("--ema_rate", type=float, default=0.9999, help="ema rate")
    group.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    group.add_argument("--log_interval", type=int, default=100, help="log interval")
    group.add_argument("--save_interval", type=int, default=25000, help="save interval")

    add_dict_to_argparser(group, diffusion_defaults())
    add_dict_to_argparser(group, diffusion_model_defaults())


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--n_samples", type=int, default=1, help="number of samples")
    group.add_argument("--input", type=str, default=None, help="input folder")
    group.add_argument("--output", type=str, default="results", help="output folder")
    group.add_argument("--resize", default=(1, 1, 1), type=float, nargs=3, help="resize factor")
    group.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim")
    group.add_argument("--timestep_respacing", type=str, default="", help="timestep respacing")
    group.add_argument("--app", type=str, default="generate", help="")
    
    group.add_argument('--reso', type=int, default=256, help="decoding volume resolution")
    group.add_argument('--n_faces', type=int, default=10000, help="number of simplified mesh faces")
    group.add_argument('--texreso', type=int, default=2048, help="texture resolution")
    group.add_argument('--vox', action='store_true', help="")
    # set default True
    group.add_argument('--copy_mtl', type=str2bool, default=True, help="copy mtl file")


def diffusion_defaults():
    return dict(
        learn_sigma=False,
        steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def diffusion_model_defaults():
    return dict(
        in_channels=12,
        model_channels=64,
        out_channels=12,
        num_res_blocks=1,
        dropout=0,
        channel_mult="1,2",
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
    )


def train_args():
    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_encoding_training_options(parser)
    add_diffusion_training_options(parser)
    args = parser.parse_args()

    # check existence
    if os.path.exists(args.tag):
        response = input(f'Folder "{args.tag}" already exists, continue? (y/n) ')
        if response != 'y':
            exit()

    os.makedirs(args.tag, exist_ok=True)
    enc_log_dir = encoding_log_dir(args.tag)
    diff_log_dir = diffusion_log_dir(args.tag)

    # encoding part
    if args.enc_log is not None: # use existing encoding, load args
        load_and_overwrite_args(args, os.path.join(args.enc_log, "args.json"))
        if not os.path.exists(enc_log_dir):
            os.symlink(os.path.abspath(args.enc_log), enc_log_dir)
    else:
        os.makedirs(enc_log_dir, exist_ok=True)
        save_path = os.path.join(enc_log_dir, "args.json")
        with open(save_path, "w") as f:
            json.dump(get_args_by_group(parser, args, "encoding"), f, indent=4)
    
    # diffusion part
    args.in_channels = args.fdim_geo if args.data_type == "sdf" else args.fdim_geo + args.fdim_tex
    args.out_channels = args.fdim_geo if args.data_type == "sdf" else args.fdim_geo + args.fdim_tex
    os.makedirs(diff_log_dir, exist_ok=True)
    save_path = os.path.join(diff_log_dir, "args.json")
    with open(save_path, "w") as f:
        json.dump(get_args_by_group(parser, args, "diffusion"), f, indent=4)
    
    # assert args.in_channels == args.out_channels == args.fdim_geo + args.fdim_tex

    # print all args
    print("----- Training args -----")
    for k, v in args.__dict__.items():
        print("{0:20}".format(k), v)

    return args


def sample_args():
    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    args = parser.parse_args()

    # print all args
    print("----- Sampling args -----")
    for k, v in args.__dict__.items():
        print("{0:20}".format(k), v)

    # check existence
    if not os.path.exists(args.tag):
        raise ValueError(f"Experiment log does not exist: {args.tag}")
    
    # load saved model args
    enc_log_dir = encoding_log_dir(args.tag)
    diff_log_dir = diffusion_log_dir(args.tag)
    load_and_overwrite_args(args, os.path.join(enc_log_dir, "args.json"))
    load_and_overwrite_args(args, os.path.join(diff_log_dir, "args.json"), ignore_keys=["timestep_respacing"])

    return args


def get_args_by_group(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return group_dict
    return ValueError('group_name was not found.')


def load_and_overwrite_args(args, path, ignore_keys=[]):
    with open(path, "r") as f:
        overwrite_args = json.load(f)
    for k, v in overwrite_args.items():
        if k not in ignore_keys:
            setattr(args, k, v)
    return args


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def encoding_log_dir(exp_tag):
    return os.path.join(exp_tag, "encoding")

    
def diffusion_log_dir(exp_tag):
    return os.path.join(exp_tag, "diffusion")


def encoding_feat_path(exp_tag):
    return os.path.join(exp_tag, "encoding/feat.npz")


def diffusion_model_path(exp_tag, ema, step):
    return os.path.join(exp_tag, f"diffusion/ema_{ema}_{step:06d}.pt")
