import os
import numpy as np
import argparse
import time
import trimesh
from trimesh.visual.color import uv_to_color
import point_cloud_utils as pcu
from utils import sample_grid_points_aabb, normalize_aabb


class MeshSampler(object):
    def __init__(self, path):
        self.path = path
        self._load(path)

    def _load(self, path):
        # load mesh obj and mtl
        scene = trimesh.load(path, force="scene", process=False)
        vs = []
        fs = []
        vns = []
        uvs = []
        tex_ids = []
        images = []
        Kas = []
        Kds = []
        Kss = []
        Nss = []

        _i = 0
        for name, geo in scene.geometry.items():
            print("load material: ", name)
            v = geo.vertices
            f = geo.faces
            vn = geo.vertex_normals
            f += sum([v.shape[0] for v in vs])
            vs.append(v)
            fs.append(f)
            vns.append(vn)
            
            uv = geo.visual.uv
            image = geo.visual.material.image

            if uv is None:
                uv = np.zeros((v.shape[0], 2)) # FIXME: this is a hack

            tex_id = np.zeros((uv.shape[0], 1), dtype=np.int32)
            tex_id[:] = _i

            Ka = np.array(geo.visual.material.ambient)[:3] / 255.0
            Kd = np.array(geo.visual.material.diffuse)[:3] / 255.0
            Ks = np.array(geo.visual.material.specular)[:3] / 255.0
            Ns = np.array(geo.visual.material.glossiness)

            uvs.append(uv)
            images.append(image)
            tex_ids.append(tex_id)
            Kas.append(Ka)
            Kds.append(Kd)
            Kss.append(Ks)
            Nss.append(Ns)
            _i += 1
        
        self.vs = np.concatenate(vs, axis=0)
        self.fs = np.concatenate(fs, axis=0)
        self.vns = np.concatenate(vns, axis=0)
        self.uvs = np.concatenate(uvs, axis=0)
        self.tex_ids = np.concatenate(tex_ids, axis=0)
        self.images = images
        self.Kas = np.array(Kas)
        self.Kds = np.array(Kds)
        self.Kss = np.array(Kss)
        self.Nss = np.array(Nss)

    def normalize(self, reso=256, enlarge_scale=1.03, mult=8):
        self.aabb, translation, scale = normalize_aabb(self.vs, reso=reso, enlarge_scale=enlarge_scale, mult=mult)
        self.vs = (self.vs + translation) * scale
        self.v_watertight = (self.v_watertight + translation) * scale

    def make_watertight_copy(self, resolution=100_000, is_watertight=False):
        if is_watertight:
            print("Watertight mesh, skipping...")
            self.v_watertight = self.vs
            self.f_watertight = self.fs
            self.n_watertight = self.vns
            return

        save_path = self.path.replace(".obj", f"_watertight_r{resolution}.obj")
        if os.path.exists(save_path):
            print("Watertight mesh exists, loading...")
            self.v_watertight, self.f_watertight, self.n_watertight = pcu.load_mesh_vfn(save_path)
            return
        start = time.time()
        self.v_watertight, self.f_watertight = pcu.make_mesh_watertight(self.vs, self.fs, resolution=resolution)
        self.n_watertight = pcu.estimate_mesh_vertex_normals(self.v_watertight, self.f_watertight)
        pcu.save_mesh_vfn(save_path, self.v_watertight, self.f_watertight, self.n_watertight)
        print("make_watertight_copy time:", time.time() - start)

    def sample_watertight_surf(self, n=1_000_000):
        f_i, bc = pcu.sample_mesh_random(self.v_watertight, self.f_watertight, n)
        surf_pts = pcu.interpolate_barycentric_coords(self.f_watertight, f_i, bc, self.v_watertight)
        return surf_pts

    def query_sdf(self, points):
        sdf, fi, bc = pcu.signed_distance_to_mesh(points, self.v_watertight, self.f_watertight)
        return sdf
    
    def query_tex(self, points):
        d, fi, bc = pcu.closest_points_on_mesh(points, self.vs, self.fs)
        tid_query = pcu.interpolate_barycentric_coords(self.fs, fi, bc, self.tex_ids.astype(np.float32))
        # fill nan with 0, FIXME: this could be a problem
        tid_query[np.isnan(tid_query)] = 0
        tid_query = tid_query.astype(np.int32)

        uv_query = pcu.interpolate_barycentric_coords(self.fs, fi, bc, self.uvs)
        uv_query[np.isnan(uv_query)] = 0

        rgba_query = self.uvs_to_colors(uv_query, tid_query)
        return rgba_query

    def uvs_to_colors(self, uvs, tex_ids):
        unique_tex_ids = np.unique(tex_ids)
        colors = np.zeros((uvs.shape[0], 4))
        for tid in unique_tex_ids:
            mask = tex_ids[..., 0] == tid
            if self.images[tid] is None: # no texture image, use diffuse color
                color = np.ones((4, ))
                color[:3] = self.Kds[tid]
                colors[mask] = color
            else:
                uv = uvs[mask]
                color = uv_to_color(uv, self.images[tid]) / 255.0
                colors[mask] = color
        return colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str)
    parser.add_argument('-d', '--dst', type=str)
    parser.add_argument('--reso', type=int, default=256)
    parser.add_argument('--watertight_reso', type=int, default=100_000)
    parser.add_argument('--n_surf', type=int, default=2_000_000)
    parser.add_argument('--mult', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--enlarge_scale', type=float, default=1.03)
    parser.add_argument('-wt', '--watertight', action='store_true')
    parser.add_argument('--only_vol', action='store_true')
    # parser.add_argument('--thres', type=int, default=0.02)
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if args.threshold is None:
        args.threshold = 2. / args.reso * 3
    print("threshold:", args.threshold)
        
    mesh = MeshSampler(src)
    mesh.make_watertight_copy(resolution=args.watertight_reso, is_watertight=args.watertight)
    mesh.normalize(reso=args.reso, enlarge_scale=args.enlarge_scale, mult=args.mult)

    # sample grid points
    vol_pts = sample_grid_points_aabb(mesh.aabb, args.reso)
    vol_shape = vol_pts.shape[:3]

    # query sdf
    vol_sdf = mesh.query_sdf(vol_pts.reshape(-1, 3))
    mask = np.abs(vol_sdf) < args.threshold
    vol_sdf = np.clip(vol_sdf, -args.threshold, args.threshold)

    # query tex
    vol_tex = np.zeros((vol_sdf.shape[0], 3))
    vol_tex[mask] = mesh.query_tex(vol_pts.reshape(-1, 3)[mask])[..., :3] # rgb

    vol_sdf = vol_sdf.reshape(vol_shape)
    vol_tex = vol_tex.reshape(vol_shape + (3,))
    mask = mask.reshape(vol_shape)
    print("vol_pts.shape:", vol_pts.shape)
    print("vol_sdf.shape:", vol_sdf.shape)
    print("vol_tex.shape:", vol_tex.shape)

    if args.only_vol:
        np.savez_compressed(dst, pts_grid=vol_pts, sdf_grid=vol_sdf, tex_grid=vol_tex,
                            aabb=mesh.aabb, threshold=args.threshold,
                            Ka=mesh.Kas[0], Kd=mesh.Kds[0], Ks=mesh.Kss[0], Ns=mesh.Nss[0])
        exit()

    # sample surface points
    on_surf_pts = mesh.sample_watertight_surf(n=args.n_surf)
    on_surf_tex = mesh.query_tex(on_surf_pts)[..., :3] # rgb
    print("on_surf_pts.shape:", on_surf_pts.shape)
    print("on_surf_tex.shape:", on_surf_tex.shape)

    # near surface points
    near_surf_pts = on_surf_pts + np.random.randn(*on_surf_pts.shape) * 0.005
    near_surf_pts = np.clip(near_surf_pts, mesh.aabb[np.newaxis, :3], mesh.aabb[np.newaxis, 3:])

    # query sdf
    near_surf_sdf = mesh.query_sdf(near_surf_pts)
    mask = np.abs(near_surf_sdf) < args.threshold
    near_surf_sdf = np.clip(near_surf_sdf, -args.threshold, args.threshold)

    # query tex
    near_surf_tex = np.zeros((near_surf_sdf.shape[0], 3))
    near_surf_tex[mask] = mesh.query_tex(near_surf_pts[mask])[..., :3] # rgb
    print("near_surf_pts.shape:", near_surf_pts.shape)
    print("near_surf_sdf.shape:", near_surf_sdf.shape)
    print("near_surf_tex.shape:", near_surf_tex.shape)

    if on_surf_pts.shape[0] > 2_000_000:
        # random downsample
        idx = np.random.choice(on_surf_pts.shape[0], 2_000_000, replace=False)
        on_surf_pts = on_surf_pts[idx]
        on_surf_tex = on_surf_tex[idx]

    np.savez_compressed(dst, pts_grid=vol_pts, sdf_grid=vol_sdf, tex_grid=vol_tex,
                        pts_on_surf=on_surf_pts, tex_on_surf=on_surf_tex,
                        pts_near_surf=near_surf_pts, sdf_near_surf=near_surf_sdf, tex_near_surf=near_surf_tex,
                        aabb=mesh.aabb, threshold=args.threshold,
                        Ka=mesh.Kas[0], Kd=mesh.Kds[0], Ks=mesh.Kss[0], Ns=mesh.Nss[0])
