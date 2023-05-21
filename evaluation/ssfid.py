import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
import random
import os
from patch_utils import load_voxgrid, load_sdfgrid2vox
from classifier3D import classifier


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(voxel, model, model_out_layer=2):
    """Calculation of the statistics used by the FID.
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    # act = get_activations(files, model, batch_size, dims, cuda, verbose)
    with torch.no_grad():
        act = model(voxel.unsqueeze(0).unsqueeze(0), out_layer=model_out_layer)
    act = act.permute(0, 2, 3, 4, 1).view(-1, act.shape[1]).detach().cpu().numpy() # (D*H*W, C)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


@torch.no_grad()
def eval_SSFID_given_paths(data_paths, ref_path, model_out_layer=2, device="cpu"):
    random.seed(1234)

    # load model
    model = classifier()
    voxel_size = 128
    weights_path = 'Clsshapenet_'+str(voxel_size)+'.pth'
    if not os.path.exists(weights_path):
        raise RuntimeError(f"'{weights_path}' not exists. Please download it from https://drive.google.com/file/d/1HjnDudrXsNY4CYhIGhH4Q0r3-NBnBaiC/view?usp=sharing.")
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    # load reference
    # gen_data_shape = load_voxgrid(data_paths[0], resolution=128, device=device).shape
    ref_data = load_sdfgrid2vox(ref_path, resolution=128, device=device).float()

    mu_r, sigma_r = calculate_activation_statistics(ref_data, model, model_out_layer)

    ssfid_values = []
    for path in tqdm(data_paths, desc="SSFID"):
        gen_data = load_voxgrid(path, resolution=128, device=device).float()

        if gen_data.shape != ref_data.shape:
            raise RuntimeError('Generated shape and reference shape shall have equal size.')
        
        mu_f, sigma_f = calculate_activation_statistics(gen_data, model, model_out_layer)
        
        ssfid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
        ssfid_values.append(ssfid)
    
    ssfid_avg = np.mean(ssfid_values).round(6)
    ssfid_std = np.std(ssfid_values).round(6)

    eval_results = {'SSFID_avg': ssfid_avg,
                    'SSFID_std': ssfid_std}
    return eval_results
