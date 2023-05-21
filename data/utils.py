import numpy as np


def sample_grid_points_aabb(aabb, resolution):
    # aabb: (6, )
    # resolution: int
    # return: (Nx, Ny, Nz, 3)
    aabb_min, aabb_max = aabb[:3], aabb[3:]
    aabb_size = aabb_max - aabb_min
    resolutions = (resolution * aabb_size / aabb_size.max()).astype(np.int32)

    xs = np.linspace(0.5, resolutions[0] - 0.5, resolutions[0]) / resolutions[0] * aabb_size[0] + aabb_min[0]
    ys = np.linspace(0.5, resolutions[1] - 0.5, resolutions[1]) / resolutions[1] * aabb_size[1] + aabb_min[1]
    zs = np.linspace(0.5, resolutions[2] - 0.5, resolutions[2]) / resolutions[2] * aabb_size[2] + aabb_min[2]
    grid_points = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    return grid_points


def normalize_aabb(v, reso, enlarge_scale=1.03, mult=8):
    aabb_min = np.min(v, axis=0)
    aabb_max = np.max(v, axis=0)
    center = (aabb_max + aabb_min) / 2
    bbox_size = (aabb_max - aabb_min).max() * enlarge_scale
    print("center:", center)
    print("bbox size", bbox_size)

    translation = -center
    scale = 1.0 / bbox_size * 2
    # v = (v + translation) * scale
    # v = (v - center) / bbox_size * 2
    aabb_min = (aabb_min * enlarge_scale - center) / bbox_size * 2
    aabb_max = (aabb_max * enlarge_scale - center) / bbox_size * 2
    aabb = np.concatenate([aabb_min, aabb_max], axis=0)
    print("v max:", v.max(axis=0), "v min:", v.min(axis=0))
    print("aabb:", aabb)

    aabb_size = aabb_max - aabb_min
    fm_size = (reso * aabb_size / aabb_size.max()).astype(np.int32)
    # round to multiple of 8
    fm_size = (fm_size + mult - 1) // mult * mult
    aabb_max = fm_size / fm_size.max()
    aabb = np.concatenate([-aabb_max, aabb_max], axis=0)
    print("aabb:", aabb)
    return aabb, translation, scale
