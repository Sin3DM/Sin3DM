import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .networks import get_networks
from .utils3d import sample_grid_points_aabb
from utils import dist_util
from utils.common_util import draw_scalar_field2D


class ShapeAutoEncoder(object):
    def __init__(self, log_dir, args):
        self.log_dir = log_dir
        
        self.batch_size = args.enc_batch_size
        self.n_iters = args.enc_n_iters
        self.vol_ratio = args.vol_ratio
        
        self.fm_reso = args.fm_reso if hasattr(args, "fm_reso") else 128
        
        self.data_type = args.data_type
        self.sdf_loss_type = args.sdf_loss
        self.tex_loss_type = args.tex_loss

        self.device = dist_util.dev()
        self.tex_weight = args.tex_weight
        self.tex_threshold_ratio = args.tex_threshold_ratio

        self.sdf_renorm = args.sdf_renorm

        self.init_lr = args.enc_lr
        self.lr_split = args.enc_lr_split if hasattr(args, "enc_lr_split") else -1
        self.min_lr_ratio = args.enc_lr_decay if hasattr(args, "enc_lr_decay") else 0.01

        # build network
        self.net = get_networks(args).to(self.device)

        # input data information
        self.Ka = None
        self.Kd = None
        self.Ks = None
        self.Ns = None
        self.aabb = None
        self.featmap_size = None
        self.sdf_threshold = None

    def _load_data(self, path, sdf_renorm=False):
        data = np.load(path)
        self.aabb = torch.from_numpy(data["aabb"]).float().to(self.device)
        self.sdf_threshold = float(data["threshold"])
        self.Ka = data["Ka"].tolist() if "Ka" in data else [0, 0, 0]
        self.Kd = data["Kd"].tolist() if "Kd" in data else [1, 1, 1]
        self.Ks = data["Ks"].tolist() if "Ks" in data else [0.4, 0.4, 0.4]
        self.Ns = data["Ns"].tolist() if "Ns" in data else 10
        print("aabb: ", self.aabb)
        print("using sdf_threshold: ", self.sdf_threshold)

        pts_grid = data["pts_grid"]
        sdf_gird = data["sdf_grid"]
        pts_near_surf = data["pts_near_surf"]
        sdf_near_surf = data["sdf_near_surf"]

        if self.data_type != "sdf":
            tex_grid = data["tex_grid"]
            pts_on_surf = data["pts_on_surf"]
            tex_on_surf = data["tex_on_surf"]
            tex_near_surf = data["tex_near_surf"]
        print("pts_grid shape: ", pts_grid.shape)
        print("pts_near_surf shape: ", pts_near_surf.shape)

        self.featmap_size = (torch.tensor(pts_grid.shape[:3]).float() * (self.fm_reso / max(pts_grid.shape[:3]))).long().tolist()
        self.featmap_size = [int(x // 2 * 2) for x in self.featmap_size]
        print("featmap size: ", self.featmap_size)

        # index volume
        if self.data_type != "sdf":
            input_grid = np.concatenate([sdf_gird[np.newaxis], tex_grid.transpose(3, 0, 1, 2)], axis=0)
        else:
            input_grid = sdf_gird[np.newaxis]
        input_grid = torch.from_numpy(input_grid).float().to(self.device)
        required_shape = [x * 2 for x in self.featmap_size]
        if input_grid.shape[1] != required_shape[0] or input_grid.shape[2] != required_shape[1] or input_grid.shape[3] != required_shape[2]:
            print("resize input_grid from ", input_grid.shape, " to ", required_shape)
            input_grid = F.interpolate(input_grid.unsqueeze(0), size=required_shape, mode="trilinear", align_corners=False).squeeze(0)
        self.input_grid = input_grid.unsqueeze(0) # [1, C, H, W, D]
        print("input grid shape: ", self.input_grid.shape)
        
        self.pts_grid = torch.from_numpy(pts_grid).float().to(self.device).view(-1, 3)
        self.sdf_grid = torch.from_numpy(sdf_gird).float().to(self.device).view(-1, 1).clamp_(-self.sdf_threshold, self.sdf_threshold)
        self.pts_near_surf = torch.from_numpy(pts_near_surf).float().to(self.device).view(-1, 3)
        self.sdf_near_surf = torch.from_numpy(sdf_near_surf).float().to(self.device).view(-1, 1).clamp_(-self.sdf_threshold, self.sdf_threshold)

        if self.data_type != "sdf":
            tex_channels = tex_grid.shape[-1]
            self.tex_grid = torch.from_numpy(tex_grid).float().to(self.device).view(-1, tex_channels)
            self.pts_on_surf = torch.from_numpy(pts_on_surf).float().to(self.device).view(-1, 3)
            self.tex_on_surf = torch.from_numpy(tex_on_surf).float().to(self.device).view(-1, tex_channels)
            self.tex_near_surf = torch.from_numpy(tex_near_surf).float().to(self.device).view(-1, tex_channels)

            if self.pts_on_surf.shape[0] > 2_000_000:
                print("downsample pts_on_surf from ", self.pts_on_surf.shape[0], " to 2,000,000")
                idx = torch.randperm(self.pts_on_surf.shape[0])[:2_000_000]
                self.pts_on_surf = self.pts_on_surf[idx]
                self.tex_on_surf = self.tex_on_surf[idx]

        if sdf_renorm:
            self.sdf_grid = self.sdf_grid / self.sdf_threshold
            self.sdf_near_surf = self.sdf_near_surf / self.sdf_threshold

    def _sample_batch(self, batch_size):
        n_grid = int(batch_size * self.vol_ratio)
        n_surf = batch_size - n_grid

        grid_idx = torch.randint(0, self.pts_grid.shape[0], (n_grid,), device=self.device)
        surf_idx = torch.randint(0, self.pts_near_surf.shape[0], (n_surf,), device=self.device)

        pts_b = torch.cat([self.pts_grid[grid_idx], self.pts_near_surf[surf_idx]], dim=0)
        sdf_b = torch.cat([self.sdf_grid[grid_idx], self.sdf_near_surf[surf_idx]], dim=0)
        if self.data_type != "sdf":
            tex_b = torch.cat([self.tex_grid[grid_idx], self.tex_near_surf[surf_idx]], dim=0)
            return {"pts": pts_b, "sdf": sdf_b, "tex": tex_b}
        else:
            return {"pts": pts_b, "sdf": sdf_b}

    def _set_optimizer(self, lr, min_lr_ratio=0.01):
        """set optimizer and lr scheduler used in training"""
        lr_decay = min_lr_ratio ** (1 / self.n_iters)
        if self.lr_split > 0:
            self.optimizer = optim.AdamW([
                {"params": self.net.geo_parameters(), "lr": lr * self.lr_split},
                {"params": self.net.tex_parameters(), "lr": lr},
            ], lr)
        else:
            self.optimizer = optim.AdamW(self.net.parameters(), lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)

    def save_ckpt(self, name):
        """save checkpoint for future restore"""
        save_path = os.path.join(self.log_dir, f"ckpt_{name}.pth")

        save_dict = {
            "net": self.net.cpu().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "Ka": self.Ka,
            "Kd": self.Kd,
            "Ks": self.Ks,
            "Ns": self.Ns,
            "aabb": self.aabb.tolist(),
            "featmap_size": self.featmap_size,
        }
        torch.save(save_dict, save_path)
        self.net.to(self.device)
    
    def load_ckpt(self, name):
        """load saved checkpoint"""
        load_path = os.path.join(self.log_dir, f"ckpt_{name}.pth")
        checkpoint = torch.load(load_path, map_location=self.device)

        self.net.load_state_dict(checkpoint["net"])
        self.Ka = checkpoint["Ka"]
        self.Kd = checkpoint["Kd"]
        self.Ks = checkpoint["Ks"]
        self.Ns = checkpoint["Ns"]
        self.aabb = torch.tensor(checkpoint["aabb"], dtype=torch.float32, device=self.device)
        self.featmap_size = checkpoint["featmap_size"]
        self.net.reset_aabb(self.aabb)

        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.net.to(self.device)

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _forward_batch(self, data):
        """forward a batch of data"""
        pts = data["pts"]
        pred = self.net(self.input_grid, pts)
        
        pred_sdf = pred[..., :1]
        gt_sdf = data["sdf"]
        if self.sdf_loss_type == "l1":
            sdf_loss = F.l1_loss(pred_sdf, gt_sdf)
        elif self.sdf_loss_type == "weightedl1":
            lamb = 0.5
            weight = (1 + lamb * torch.sign(gt_sdf) * torch.sign(gt_sdf - pred_sdf))
            sdf_loss = ((pred_sdf - gt_sdf).abs() * weight).mean()
        elif self.sdf_loss_type == "weightedl1_clamp":
            lamb = 0.5
            gt_sdf = torch.clamp(gt_sdf, -self.sdf_threshold, self.sdf_threshold)
            pred_sdf = torch.clamp(pred_sdf, -self.sdf_threshold, self.sdf_threshold)
            weight = (1 + lamb * torch.sign(gt_sdf) * torch.sign(gt_sdf - pred_sdf))
            sdf_loss = ((pred_sdf - gt_sdf).abs() * weight).mean()
        else:
            raise NotImplementedError
        loss_dict = {"sdf_loss": sdf_loss}
        
        if self.data_type != "sdf":
            pred_tex = pred[..., 1:]
            gt_tex = data["tex"]
            if self.sdf_renorm:
                mask = gt_sdf.squeeze(1).abs() < 1.0 * self.tex_threshold_ratio
            else:
                mask = gt_sdf.squeeze(1).abs() < self.sdf_threshold * self.tex_threshold_ratio
            if self.data_type == "sdftex":
                if self.tex_loss_type == "l1":
                    tex_loss = F.l1_loss(pred_tex[mask], gt_tex[mask]) * self.tex_weight
                elif self.tex_loss_type == "l2":
                    tex_loss = F.mse_loss(pred_tex[mask], gt_tex[mask]) * self.tex_weight
                elif self.tex_loss_type == "huber":
                    tex_loss = F.huber_loss(pred_tex[mask], gt_tex[mask], delta=0.1) * self.tex_weight
                else:
                    raise NotImplementedError
                loss_dict["tex_loss"] = tex_loss
            elif self.data_type == "sdfpbr":
                if self.tex_loss_type == "l1":
                    rgb_loss = F.l1_loss(pred_tex[mask, :3], gt_tex[mask, :3]) * self.tex_weight
                    mr_loss = F.l1_loss(pred_tex[mask, 3:5], gt_tex[mask, 3:5]) * self.tex_weight
                    normal_loss = F.l1_loss(pred_tex[mask, 5:], gt_tex[mask, 5:]) * self.tex_weight
                    loss_dict["rgb_loss"] = rgb_loss
                    loss_dict["mr_loss"] = mr_loss
                    loss_dict["normal_loss"] = normal_loss
            else:
                raise NotImplementedError

        return pred, loss_dict

    def train(self, data_path):
        # load data
        self._load_data(data_path, sdf_renorm=self.sdf_renorm)
        self.net.reset_aabb(self.aabb)
        
        # set optimizer
        self._set_optimizer(self.init_lr, self.min_lr_ratio)

        # set tensorboard writer
        self.tb = SummaryWriter(os.path.join(self.log_dir, "tblog"))

        pbar = tqdm(range(self.n_iters))
        self.step = 0
        for i in pbar:
            self.step = i
            self.net.train()

            data = self._sample_batch(self.batch_size)

            rec, loss_dict = self._forward_batch(data)

            self.update_network(loss_dict)
            
            loss_values = {k: v.item() for k, v in loss_dict.items()}
            self.tb.add_scalars("loss", loss_values, global_step=i)
            pbar.set_postfix(loss_values)

            if i == 0 or (i + 1) % (self.n_iters // 5) == 0:
                self._visualize_batch(i)

            if (i + 1) % (self.n_iters // 5) == 0:
                eval_stat = self.evaluate()
                for name in ["tsdf_l1", "tsdf_rel", "tsdf_acc"]:
                    stat_dict = {k: v for k, v in eval_stat.items() if name in k}
                    self.tb.add_scalars(name, stat_dict, global_step=i)
                if self.data_type != "sdf":
                    self.tb.add_scalar("surf_tex_l1_error", eval_stat["surf_tex_l1_error"], global_step=i)

        eval_stat = self.evaluate()
        with open(os.path.join(self.log_dir, "eval_stat.json"), "w") as f:
            json.dump(eval_stat, f, indent=2)
        self.save_ckpt("final")

    @torch.no_grad()
    def _visualize_batch(self, step):
        bs = 2
        feat_maps = self.encode()
        for i in range(3):
            fm = feat_maps[i].detach().cpu().numpy()[0, 0]
            self.tb.add_figure(f"feat_map_{i}", draw_scalar_field2D(fm), global_step=step)
    
    @torch.no_grad()
    def evaluate(self):
        self.net.eval()

        feat_maps = self.encode()
        sdf_grid_pred = self.decode_batch(feat_maps, self.pts_grid)[..., :1]

        if self.sdf_renorm:
            sdf_grid_pred = sdf_grid_pred * self.sdf_threshold
            sdf_grid_gt = self.sdf_grid * self.sdf_threshold
        else:
            sdf_grid_gt = self.sdf_grid
        stat = evaluate_tsdf_prediction(sdf_grid_pred, sdf_grid_gt, self.sdf_threshold)

        if self.data_type != "sdf":
            tex_surf_pred = self.decode_batch(feat_maps, self.pts_on_surf)[..., 1:] # (n, 3)
            surf_tex_l1_error = (tex_surf_pred - self.tex_on_surf).abs().mean().item()
            stat["surf_tex_l1_error"] = surf_tex_l1_error

        return stat

    @torch.no_grad()
    def encode(self, vol=None):
        """encode a patch of volume"""
        if vol is None:
            vol = self.input_grid # (1, C, H, W, D)
        self.net.eval()
        return self.net.encode(vol)

    @torch.no_grad()
    def decode_batch(self, triplane_feat, points, batch_size=2 ** 14, aabb=None):
        """decode a batch of points
        triplane_feat: (c, h, w, d)
        """
        self.net.eval()

        preds = []
        for i in tqdm(range(0, points.shape[0], batch_size)):
            pts = points[i:i+batch_size]
            rec = self.net.decode(pts, triplane_feat, aabb=aabb)
            preds.append(rec)
        preds = torch.cat(preds, dim=0)
        preds[..., 1:] = preds[..., 1:].clamp_(0, 1) # color in [0, 1]
        return preds
    
    @torch.no_grad()
    def decode_grid(self, triplane_feat, reso, batch_size=2 ** 14, aabb=None):
        """decode feature volume at grid points"""
        self.net.eval()

        # coords = sample_uniform(reso, sdim=3, device=self.device) # (H, W, D, 3)
        if aabb is None:
            aabb = self.aabb
        coords = sample_grid_points_aabb(aabb, reso) # (H, W, D, 3)
        H, W, D, _ = coords.shape
        coords_list = coords.view(-1, 3) # (H*W*D, 3)

        preds = self.decode_batch(triplane_feat, coords_list, batch_size=batch_size, aabb=aabb)
        preds = preds.view(H, W, D, -1)
        return preds

    def _resize_aabb(self, featmap_size):
        if featmap_size[0] != self.featmap_size[0] or featmap_size[1] != self.featmap_size[1] or featmap_size[2] != self.featmap_size[2]:
            scale = torch.tensor([featmap_size[0] / self.featmap_size[0], featmap_size[1] / self.featmap_size[1], featmap_size[2] / self.featmap_size[2]], device=self.device)
            new_aabb = self.aabb.clone()
            new_aabb[:3] = self.aabb[:3] * scale
            new_aabb[3:] = self.aabb[3:] * scale
            print("resized aabb:", new_aabb)
            return new_aabb
        else:
            return self.aabb

    @torch.no_grad()
    def decode_texmesh(self, save_dir, triplane_feat, reso, n_faces=10000, n_surf_pc=-1, texture_reso=2048, only_largest_cc=True, 
                       save_highres_mesh=False, save_voxel=True, mtl_path=None, file_format="obj"):
        from .utils3d import sdfgrid_to_mesh, xatlas_uvmap, save_mesh_with_tex, save_mesh_with_tex_to_glb, mesh_decimation, save_mesh_with_pbr, read_metarial_params_from_mtl
        import point_cloud_utils as pcu
        import cv2

        H, W = triplane_feat[0].shape[-2:]
        D = triplane_feat[1].shape[-1]
        new_aabb = self._resize_aabb((H, W, D))

        # maching cubes on sdf grid
        os.makedirs(save_dir, exist_ok=True)
        sdf_grid = self.decode_grid(triplane_feat, reso, aabb=new_aabb)[..., 0].detach().cpu().numpy() # (H, W, D, 1)
        if save_voxel:
            vox_grid = sdf_grid < 0
            save_path = os.path.join(save_dir, f"voxel.npz")
            np.savez_compressed(save_path, vox_grid=vox_grid)

        save_path = os.path.join(save_dir, f"mesh_r{reso}.obj") if save_highres_mesh else None
        v, f = sdfgrid_to_mesh(sdf_grid, save_path, only_largest_cc=only_largest_cc)

        # re normalize
        box_min = new_aabb[:3].detach().cpu().numpy()
        box_size = new_aabb[3:].max().item() - new_aabb[:3].min().item()
        v = v / reso * box_size + box_min

        # decimation
        v, f = mesh_decimation(v, f, n_faces)

        if not self.data_type != "sdf":
            save_path = os.path.join(save_dir, f"sdfgrid_r{reso}.npz")
            np.savez_compressed(save_path, sdf_grid=sdf_grid)
            save_path = os.path.join(save_dir, f"mesh_r{reso}_simple.obj")
            pcu.save_mesh_vf(save_path, v, f)
            return

        # also save surface point cloud
        if n_surf_pc > 0:
            f_i, bc = pcu.sample_mesh_random(v, f, n_surf_pc)
            surf_points = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

            coords = torch.from_numpy(surf_points).float().to(self.device)
            preds = self.decode_batch(triplane_feat, coords, aabb=new_aabb)
            save_path = os.path.join(save_dir, f"surf_pc_n{n_surf_pc}.obj")
            coords = coords.detach().cpu().numpy()
            colors = preds[..., 1:4].detach().cpu().numpy()
            colors = np.clip(colors, 0, 1)
            pcu.save_mesh_vc(save_path, coords, colors)

        # uv map
        v = torch.from_numpy(v).float().to(self.device)
        f = torch.from_numpy(f.astype(int)).long().to(self.device)
        uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(v, f, texture_reso)

        preds = self.decode_batch(triplane_feat, gb_pos.view(-1, 3)[mask.view(-1)], aabb=new_aabb)
        tex_img = torch.zeros((texture_reso, texture_reso, preds.shape[-1] - 1), device=preds.device)
        tex_img[mask.view(texture_reso, texture_reso)] = preds[..., 1:].clamp_(0, 1)
        # tex_img = preds[..., 1:].clamp_(0, 1).view(texture_reso, texture_reso, 3).detach().cpu().numpy()
        tex_img = tex_img.detach().cpu().numpy()
        tex_img = (tex_img * 255).astype(np.uint8)

        mask = mask.view(texture_reso, texture_reso, 1).detach().cpu().numpy()

        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(tex_img, kernel, iterations=1)
        tex_img = tex_img * mask + dilate_img * (1 - mask)
        tex_img = tex_img.clip(0, 255).astype(np.uint8)
        tex_img = tex_img[::-1] # flip

        if self.data_type == "sdftex":
            if file_format == "obj":
                save_path = os.path.join(save_dir, "object.obj")
                mtl_str = read_metarial_params_from_mtl(mtl_path) if mtl_path is not None else None
                save_mesh_with_tex(
                    save_path,
                    v.detach().cpu().numpy(),
                    uvs.detach().cpu().numpy(),
                    f.detach().cpu().numpy(),
                    mesh_tex_idx.detach().cpu().numpy(),
                    tex_img, 
                    mtl_str=mtl_str,
                    Kd=self.Kd, Ka=self.Ka, Ks=self.Ks, Ns=self.Ns
                )
            elif file_format == "glb":
                save_path = os.path.join(save_dir, "object.glb")
                save_mesh_with_tex_to_glb(
                    save_path.replace(".obj", ".glb"),
                    v.detach().cpu().numpy(),
                    uvs.detach().cpu().numpy(),
                    f.detach().cpu().numpy(),
                    mesh_tex_idx.detach().cpu().numpy(),
                    tex_img
                )
            else:
                raise NotImplementedError

        elif self.data_type == "sdfpbr":
            albedo_img = tex_img[..., :3]
            metallic_img = tex_img[..., 3]
            roughness_img = tex_img[..., 4]
            normal_img = tex_img[..., 5:]
            save_mesh_with_pbr(
                save_path,
                v.detach().cpu().numpy(),
                uvs.detach().cpu().numpy(),
                f.detach().cpu().numpy(),
                mesh_tex_idx.detach().cpu().numpy(),
                albedo_img, metallic_img, roughness_img, normal_img
            )
        else:
            raise NotImplementedError

    @torch.no_grad()
    def decode_voxel(self, save_dir, triplane_feat, reso, n_faces=10000, only_largest_cc=True):
        H, W = triplane_feat[0].shape[-2:]
        D = triplane_feat[1].shape[-1]
        new_aabb = self._resize_aabb((H, W, D))

        # maching cubes on sdf grid
        os.makedirs(save_dir, exist_ok=True)
        sdf_grid = self.decode_grid(triplane_feat, reso, aabb=new_aabb)[..., 0].detach().cpu().numpy() # (H, W, D, 1)
        # sdf_grid = -F.max_pool3d(-sdf_grid[None, None], 2, 2)[0, 0].detach().cpu().numpy()
        vox_grid = sdf_grid < 0

        save_path = os.path.join(save_dir, f"r{reso}_voxel.npz")
        np.savez_compressed(save_path, vox_grid=vox_grid)


def evaluate_tsdf_prediction(pred_sdf, gt_sdf, sdf_threshold):
    res = {}

    l1_error = torch.abs(pred_sdf - gt_sdf)
    rel_error = l1_error / torch.abs(gt_sdf)
    acc = (pred_sdf * gt_sdf >= 0).float()

    res["mean_tsdf_l1_error"] = l1_error.mean().item()
    res["mean_tsdf_rel_error"] = rel_error.mean().item()
    res["mean_tsdf_acc"] = acc.mean().item()
    
    n = 4
    unit = sdf_threshold / n
    threshold_ranges = [i * unit for i in range(n + 1)] + [unit * (n + 1)]

    for i in range(len(threshold_ranges) - 1):
        lower = threshold_ranges[i]
        upper = threshold_ranges[i + 1]
        mask = (gt_sdf.abs() >= lower) & (gt_sdf.abs() < upper)

        res[f"mean_tsdf_l1_error_{i}-{n}-{i + 1}-n"] = l1_error[mask].mean().item()
        res[f"mean_tsdf_rel_error_{i}-{n}-{i + 1}-n"] = rel_error[mask].mean().item()
        res[f"mean_tsdf_acc_{i}-{n}-{i + 1}-n"] = acc[mask].mean().item()
        res[f"mean_tsdf_count_{i}-{n}-{i + 1}-n"] = mask.sum().item()

    return res
