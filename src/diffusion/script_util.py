from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet_triplane import TriplaneUNetModelSmall, TriplaneUNetModelSmallRaw
from utils.parser_util import diffusion_defaults, diffusion_model_defaults, args_to_dict


def create_model_and_diffusion_from_args(args):
    """
    Create model and diffusion from args.
    """
    diffusion = create_gaussian_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    
    if type(args.channel_mult) is str:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))
    if args.diff_net_type == "unet_small":
        model = TriplaneUNetModelSmall(**args_to_dict(args, diffusion_model_defaults().keys()))
    elif args.diff_net_type == "unet_raw":
        model = TriplaneUNetModelSmallRaw(**args_to_dict(args, diffusion_model_defaults().keys()))
    return model, diffusion


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
