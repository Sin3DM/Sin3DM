import open3d as o3d
import numpy as np
import xatlas
import nvdiffrast.torch as dr
import torch
import PIL
import os
import mcubes
import point_cloud_utils as pcu
import trimesh


def sample_grid_points_aabb(aabb, resolution):
    # aabb: (6, )
    # resolution: int
    # return: (Nx, Ny, Nz, 3)
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    aabb_size = aabb_max - aabb_min
    resolutions = (resolution * aabb_size / aabb_size.max()).long()

    xs = torch.linspace(0.5, resolutions[0] - 0.5, resolutions[0], device=aabb.device) / resolutions[0] * aabb_size[0] + aabb_min[0]
    ys = torch.linspace(0.5, resolutions[1] - 0.5, resolutions[1], device=aabb.device) / resolutions[1] * aabb_size[1] + aabb_min[1]
    zs = torch.linspace(0.5, resolutions[2] - 0.5, resolutions[2], device=aabb.device) / resolutions[2] * aabb_size[2] + aabb_min[2]
    grid_points = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
    return grid_points


def read_metarial_params_from_mtl(path):
    with open(path, "r") as f:
        lines = f.readlines()

    s = ""
    start = False
    for l in lines:
        _l = l.lstrip()
        if start is False and _l[:6] == "newmtl":
            start = True
            continue
        if _l[:4] == "map_" or _l[:6] == "newmtl":
            break
        if start:
            s += l
    return s


def save_mesh_with_tex(fname, pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, tex_img,
                       mtl_str=None, Kd=[1, 1, 1], Ka=[0, 0, 0], Ks=[0.4, 0.4, 0.4], Ns=10, illum=2):
    assert fname.endswith('.obj')

    na = os.path.basename(fname)[:-4]

    matname = fname.replace('.obj', '.mtl')
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    if mtl_str is not None:
        fid.write(mtl_str)
    else:
        fid.write(f'Kd {Kd[0]} {Kd[1]} {Kd[2]}\n')
        fid.write(f'Ka {Ka[0]} {Ka[1]} {Ka[2]}\n')
        fid.write(f'Ks {Ks[0]} {Ks[1]} {Ks[2]}\n')
        fid.write(f'Ns {Ns}\n')
        fid.write(f'illum {illum}\n')
    fid.write('map_Kd %s.png\n' % na)
    fid.close()

    imgname = fname.replace('.obj', '.png')
    PIL.Image.fromarray(tex_img).save(imgname)
    ####

    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    return


def save_mesh_with_tex_to_glb(fname, pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, tex_img):
    assert fname.endswith('.glb')
    v_new = []
    vt_new = []
    f_new = []
    v2idx_dict = {}
    for i in range(len(facenp_fx3)):
        f1 = facenp_fx3[i]
        f2 = facetex_fx3[i]
        f_new_idx = []
        for j in range(3):
            v_unique_id = str(f1[j]) + '_' + str(f2[j])
            if v_unique_id not in v2idx_dict.keys():
                v2idx_dict[v_unique_id] = len(v_new)
                v_new.append(pointnp_px3[f1[j]])
                vt_new.append(tcoords_px2[f2[j]])
            f_new_idx.append(v2idx_dict[v_unique_id])
        f_new.append(f_new_idx)
    
    v_new = np.array(v_new)
    vt_new = np.array(vt_new)
    f_new = np.array(f_new)
    # ft_new = f_new.copy()

    visual = trimesh.visual.texture.TextureVisuals(
        uv=vt_new,
        image=PIL.Image.fromarray(tex_img),
    )

    mesh = trimesh.Trimesh(
        vertices=v_new,
        faces=f_new,
        visual=visual,
    )

    def tree_postprocessor(tree):
        # FIXME: hardcode the material
        tree["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        tree["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] = 0.0
        tree["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
        tree["materials"][0]["doubleSided"] = True

    with open(fname, 'wb') as f:
        mesh.export(file_obj=f, file_type='glb', tree_postprocessor=tree_postprocessor)


def save_mesh_with_pbr(fname, pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, 
                       albedo_img, metallic_img, roughness_img, normal_img,
                       Ks=[0.5, 0.5, 0.5], Ke=[0, 0, 0], Ns=250, Ni=1.5, d=1.0, illum=2,
                       Ps=0.0, Pc=0.0, Pcr=0.03, aniso=0.0, anisor=0.0):
    assert fname.endswith('.obj')

    na = os.path.basename(fname)[:-4]
    tex_dir = os.path.join(os.path.dirname(fname), 'textures')
    os.makedirs(tex_dir, exist_ok=True)

    matname = fname.replace('.obj', '.mtl')
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write(f'Ns {Ns}\n')
    fid.write(f'Ks {Ks[0]} {Ks[1]} {Ks[2]}\n')
    fid.write(f'Ke {Ke[0]} {Ke[1]} {Ke[2]}\n')
    fid.write(f'Ni {Ni}\n')
    fid.write(f'd {d}\n')
    fid.write(f'illum {illum}\n')
    fid.write(f'Ps {Ps}\n')
    fid.write(f'Pc {Pc}\n')
    fid.write(f'Pcr {Pcr}\n')
    fid.write(f'aniso {aniso}\n')
    fid.write(f'anisor {anisor}\n')

    fid.write(f'map_Kd textures/albedo.png\n')
    fid.write(f'map_Pm textures/metallic.png\n')
    fid.write(f'map_Pr textures/roughness.png\n')
    fid.write(f'map_Bump -bm 1.000000 textures/normal.png\n')
    fid.close()

    # save texture images
    PIL.Image.fromarray(albedo_img).save(os.path.join(tex_dir, 'albedo.png'))
    PIL.Image.fromarray(metallic_img).save(os.path.join(tex_dir, 'metallic.png'))
    PIL.Image.fromarray(roughness_img).save(os.path.join(tex_dir, 'roughness.png'))
    PIL.Image.fromarray(normal_img).save(os.path.join(tex_dir, 'normal.png'))

    ####
    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    return


def sdfgrid_to_mesh(sdf_grid, save_path=None, only_largest_cc=True, is_voxel=False):
    if is_voxel:
        sdf_grid = np.pad(sdf_grid, 1, mode='constant', constant_values=0)
        v, f = mcubes.marching_cubes(sdf_grid, 0.5)
    else:
        sdf_grid = np.pad(sdf_grid, 1, mode='constant', constant_values=1.0)
        v, f = mcubes.marching_cubes(sdf_grid, 0)
    v -= 1.0 # Remove padding
    if only_largest_cc:
        f = f.astype(np.int64)
        cv, nv, cf, nf = pcu.connected_components(v, f)
        # Extract mesh of connected component with most faces
        comp_max = np.argmax(nf)
        v, f, _, _ = pcu.remove_unreferenced_mesh_vertices(v, f[cf == comp_max])

    if save_path is not None:
        mcubes.export_obj(v, f, save_path)
    return v, f


def mesh_decimation(v, f, face_count=10000):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh = mesh.simplify_quadric_decimation(face_count)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def xatlas_uvmap(mesh_v, mesh_f, resolution):
    ctx = dr.RasterizeGLContext(device=mesh_v.device)
    vmapping, indices, uvs = xatlas.parametrize(mesh_v.detach().cpu().numpy(), mesh_f.detach().cpu().numpy())
    vmapping = torch.tensor(vmapping.astype(np.int64), dtype=torch.int64, device=mesh_v.device)
    # mesh_v = mesh_v[vmapping]

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
    mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
    # mesh_v_tex. ture
    uv_clip = uvs[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), (resolution, resolution))

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_f.int())
    mask = rast[..., 3:4] > 0
    return uvs, mesh_tex_idx, gb_pos, mask
