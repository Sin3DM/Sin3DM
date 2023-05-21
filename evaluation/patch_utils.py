import numpy as np
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm


def load_sdfgrid2vox(path, binarize=True, resolution=128, device="cpu"):
    sdfgrid = np.load(path)["sdf_grid"][:]
    sdfgrid = torch.from_numpy(sdfgrid).float().to(device)

    if max(sdfgrid.shape) != resolution:
        new_shape = [int(x * resolution / max(sdfgrid.shape)) for x in sdfgrid.shape]
        sdfgrid = -F.adaptive_max_pool3d(-sdfgrid[None, None], new_shape)[0, 0]

    if binarize:
        sdfgrid = sdfgrid <= 0
    return sdfgrid


def load_voxgrid(path, resolution=128, device="cpu"):
    voxgrid = np.load(path)["vox_grid"][:]
    voxgrid = torch.from_numpy(voxgrid).bool().to(device)
    if max(voxgrid.shape) != resolution:
        new_shape = [int(x * resolution / max(voxgrid.shape)) for x in voxgrid.shape]
        voxgrid = F.adaptive_max_pool3d(voxgrid[None, None].float(), new_shape)[0, 0].bool()
    return voxgrid


def pairwise_IoU_dist(data_list: list):
    """average pairwise 1-IoU for a list of 3D shape volume"""
    avgv = []
    for i in tqdm(range(len(data_list)), desc='Div'):
        data_i = data_list[i]
        intersect = torch.logical_and(data_i, data_list).sum(dim=(1, 2, 3))
        union = torch.logical_or(data_i, data_list).sum(dim=(1, 2, 3))
        iou_dist = 1.0 - intersect / union
        mask = torch.ones_like(iou_dist, dtype=torch.bool)
        mask[i] = False
        iou_dist = iou_dist[mask]
        avgv.append(torch.mean(iou_dist).item())
    avgv = np.mean(avgv)
    return avgv


def extract_valid_patches_unfold(voxels: torch.Tensor, patch_size: int, stride=None):
    """extract near-surface patches of a 3D shape using torch.unfold

    Args:
        voxels (torch.Tensor): a 3D shape volume of size (H, W, D)
        patch_size (int): patch size
        stride (int, optional): stride for overlapping. Defaults to None. If None, set as half patch size.

    Returns:
        patches: size (N, patch_size, patch_size, patch_size)
    """
    overlap = patch_size // 2 if stride is None else stride

    p = patch_size // 2
    voxels = F.pad(voxels, [p, p, p, p, p, p])
    patches = voxels.unfold(0, patch_size, overlap).unfold(1, patch_size, overlap).unfold(2, patch_size, overlap) 
    patches = patches.contiguous().view(-1, patch_size, patch_size, patch_size) # (k, ps, ps, ps)
    
    # valid patch criterion
    # center region (l^3) has at least one occupied and one unoccupied voxel
    idx = patch_size // 2 - 1
    l = 2 if patch_size % 2 == 0 else 3
    centers = patches[:, idx:idx+l, idx:idx+l, idx:idx+l] # (k, l, l, l)
    mask_occ = torch.sum(centers.int(), dim=(1, 2, 3)) > 0 # (k,)
    mask_unocc = torch.sum(centers.int(), dim=(1, 2, 3)) < l * l * l # (k,)
    mask = torch.logical_and(mask_occ, mask_unocc)

    patches = patches[mask]
    return patches


def eval_LP_IoU(gen_patches: torch.Tensor, ref_patches: torch.Tensor, threshold=0.95):
    """compute LP-IoU over two set of patches.

    Args:
        gen_patches (torch.Tensor): patches from generated shape
        ref_patches (torch.Tensor): patches from reference shape
        threshold (float, optional): IoU threshold. Defaults to 0.95.

    Returns:
        average max IoU, LP-IoU
    """
    values = []
    for i in range(gen_patches.shape[0]):
        intersect = torch.logical_and(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        union = torch.logical_or(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        max_iou = torch.max(intersect / union)
        values.append(max_iou)
    values = torch.stack(values)
    avg_iou = torch.mean(values).item()
    percent = torch.sum((values > threshold).int()).item() * 1.0 / len(values)
    return avg_iou, percent


def eval_LP_Fscore(gen_patches: torch.Tensor, ref_patches: torch.Tensor, threshold=0.95):
    """compute LP-F-score over two set of patches.

    Args:
        gen_patches (torch.Tensor): patches from generated shape
        ref_patches (torch.Tensor): patches from reference shape
        threshold (float, optional): F-score threshold. Defaults to 0.95.

    Returns:
        average max F-score, LP-F-score
    """
    values = []
    for i in range(gen_patches.shape[0]):
        true_positives = torch.logical_and(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        precision = true_positives / gen_patches[i:i+1].sum()
        recall = true_positives / ref_patches.sum(dim=(1, 2, 3))
        Fscores = 2 * precision * recall / (precision + recall + 1e-8)
        Fscore = torch.max(Fscores)
        values.append(Fscore)
    values = torch.stack(values)
    avg_fscore = torch.mean(values).item()
    percent = torch.sum((values > threshold).int()).item() * 1.0 / len(values)
    return avg_fscore, percent


def eval_LP_given_paths(data_paths, ref_path, patch_size=11, stride=5, patch_num=1000, device="cpu"):
    random.seed(1234)

    # load reference
    gen_data_shape = load_voxgrid(data_paths[0], resolution=128, device=device).shape
    ref_data = load_sdfgrid2vox(ref_path, resolution=128, device=device)

    ref_patches = extract_valid_patches_unfold(ref_data, patch_size, stride)
    
    # LP
    result_lp_iou_avg = []
    result_lp_iou_percent = []
    result_lp_fscore_avg = []
    result_lp_fscore_percent = []

    for path in tqdm(data_paths, desc="LP-IOU/F-score"):
        gen_data = load_voxgrid(path, resolution=128, device=device)

        gen_patches = extract_valid_patches_unfold(gen_data, patch_size, stride)
        indices = list(range(gen_patches.shape[0]))
        random.shuffle(indices)
        indices = indices[:patch_num]
        gen_patches = gen_patches[indices]

        lp_iou_avg, lp_iou_percent = eval_LP_IoU(gen_patches, ref_patches)
        lp_fscore_avg, lp_fscore_percent = eval_LP_Fscore(gen_patches, ref_patches)

        result_lp_iou_avg.append(lp_iou_avg)
        result_lp_iou_percent.append(lp_iou_percent)
        result_lp_fscore_avg.append(lp_fscore_avg)
        result_lp_fscore_percent.append(lp_fscore_percent)

    result_lp_iou_avg = np.mean(result_lp_iou_avg).round(6)
    result_lp_fscore_avg = np.mean(result_lp_fscore_avg).round(6)
    result_lp_iou_percent = np.mean(result_lp_iou_percent).round(6)
    result_lp_fscore_percent = np.mean(result_lp_fscore_percent).round(6)
    
    eval_results = {'LP-IOU-avg': result_lp_iou_avg,
                    'LP-IOU-percent': result_lp_iou_percent,
                    'LP-F-score-avg': result_lp_fscore_avg,
                    'LP-F-score-percent': result_lp_fscore_percent}
    return eval_results


def eval_Div_given_paths(data_paths, device="cpu"):
    random.seed(1234)

    gen_data_list = []
    for path in data_paths:
        gen_data = load_voxgrid(path, resolution=128, device=device)
        gen_data_list.append(gen_data)

    gen_data_list = torch.stack(gen_data_list, dim=0)
    div = pairwise_IoU_dist(gen_data_list).round(6)
    
    eval_results = {'Div': div}
    return eval_results
