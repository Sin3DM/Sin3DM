import torch
import torch.nn.functional as F
import numpy as np
import os


def compose_featmaps(feat_xy, feat_xz, feat_yz):
    H, W = feat_xy.shape[-2:]
    D = feat_xz.shape[-1]

    empty_block = torch.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    composed_map = torch.cat(
        [torch.cat([feat_xy, feat_xz], dim=-1),
         torch.cat([feat_yz.transpose(-1, -2), empty_block], dim=-1)], 
        dim=-2
    )
    return composed_map, (H, W, D)


def decompose_featmaps(composed_map, sizes):
    H, W, D = sizes
    feat_xy = composed_map[..., :H, :W] # (C, H, W)
    feat_xz = composed_map[..., :H, W:] # (C, H, D)
    feat_yz = composed_map[..., H:, :W].transpose(-1, -2) # (C, W, D)
    return feat_xy, feat_xz, feat_yz


def pad_composed_featmaps(composed_map, sizes, pad_sizes):
    # pad_sizes: [[padH1, padH2], [padW1, padW2], [padD1, padD2]]
    feat_xy, feat_xz, feat_yz = decompose_featmaps(composed_map, sizes)
    feat_xy = F.pad(feat_xy, pad_sizes[1] + pad_sizes[0])
    feat_xz = F.pad(feat_xz, pad_sizes[2] + pad_sizes[0])
    feat_yz = F.pad(feat_yz, pad_sizes[2] + pad_sizes[1])
    composed_map, new_sizes = compose_featmaps(feat_xy, feat_xz, feat_yz)
    return composed_map, new_sizes


def save_triplane_data(path, feat_xy, feat_xz, feat_yz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, 
                        feat_xy=feat_xy, feat_xz=feat_xz, feat_yz=feat_yz)


def load_triplane_data(path, device="cuda:0", compose=True):
    data = np.load(path)
    feat_xy = data['feat_xy'][:]
    feat_xz = data['feat_xz'][:]
    feat_yz = data['feat_yz'][:]
    # print("feat_xy shape:", feat_xy.shape)
    # print("feat_xz shape:", feat_xz.shape)
    # print("feat_yz shape:", feat_yz.shape)

    feat_xy = torch.from_numpy(feat_xy).float().to(device) # (C, H, W)
    feat_xz = torch.from_numpy(feat_xz).float().to(device) # (C, H, D)
    feat_yz = torch.from_numpy(feat_yz).float().to(device) # (C, W, D)

    if not compose:
        return feat_xy, feat_xz, feat_yz

    composed_map, (H, W, D) = compose_featmaps(feat_xy, feat_xz, feat_yz)
    return composed_map, (H, W, D)


def get_data_iterator(featmaps_data, sizes, batch_size=1):
    featmaps_data = featmaps_data.unsqueeze(0).expand(batch_size, -1, -1, -1)
    H, W, D = sizes

    while True:
        yield featmaps_data, {'H': H, 'W': W, 'D': D}
