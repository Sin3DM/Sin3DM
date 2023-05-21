import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DecoderMLP, ResnetBlock, TriplaneGroupResnetBlock, DecoderMLPSkipConcat


def get_networks(cfg):
    print("Encoding network type: {}".format(cfg.enc_net_type))
    use_tex = cfg.data_type != "sdf"
    tex_channels = 8 if cfg.data_type == "sdfpbr" else 3
    if cfg.enc_net_type == "base":
        return AutoEncoderGroupV3(cfg.fdim_geo, cfg.fdim_tex, cfg.fdim_up, cfg.hidden_dim, cfg.n_hidden_layers, use_tex=use_tex, tex_channels=tex_channels)
    elif cfg.enc_net_type == "skip":
        return AutoEncoderGroupSkip(cfg.fdim_geo, cfg.fdim_tex, cfg.fdim_up, cfg.hidden_dim, cfg.n_hidden_layers, use_tex=use_tex, tex_channels=tex_channels)
    elif cfg.enc_net_type == "pbr":
        return AutoEncoderGroupPBR(cfg.fdim_geo, cfg.fdim_tex, cfg.fdim_up, cfg.hidden_dim, cfg.n_hidden_layers, use_tex=use_tex, tex_channels=tex_channels)
    else:
        raise ValueError("Unknown net type: {}".format(cfg.net_type))


class AutoEncoderGroupV3(nn.Module):
    def __init__(self, geo_feat_channels, tex_feat_channels, feat_channel_up, mlp_hidden_channels, mlp_hidden_layers, use_tex=True, tex_channels=3, posenc=0) -> None:
        super().__init__()
        self.use_tex = use_tex

        self.geo_encoder = nn.Conv3d(1, geo_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        if use_tex:
            self.tex_encoder = nn.Conv3d(tex_channels + 1, tex_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        out_channels = geo_feat_channels + tex_feat_channels if use_tex else geo_feat_channels
        self.norm = nn.InstanceNorm2d(out_channels)

        self.geo_feat_dim = geo_feat_channels
        self.tex_feat_dim = tex_feat_channels

        self.geo_convs = TriplaneGroupResnetBlock(
            geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
        )
        self.geo_decoder = DecoderMLP(feat_channel_up, 1, mlp_hidden_channels, mlp_hidden_layers)

        if use_tex:
            self.tex_convs = TriplaneGroupResnetBlock(
                tex_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
            )
            self.tex_decoder = DecoderMLP(feat_channel_up, tex_channels, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
        
        self.register_buffer("aabb", torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32))

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + list(self.tex_decoder.parameters())

    def reset_aabb(self, aabb):
        print("set net aabb:", aabb)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # self.register_buffer("aabb", aabb.to(self.encoder.weight.device))
        self.aabb = aabb.to(self.geo_encoder.weight.device)

    def encode(self, vol):
        geo_feat = self.geo_encoder(vol[:, :1])
        if self.use_tex:
            tex_feat = self.tex_encoder(vol)
            vol_feat = torch.cat([geo_feat, tex_feat], dim=1)
        else:
            vol_feat = geo_feat

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def decode(self, x, feat_maps, aabb=None):
        # x [N, 3]
        if aabb is None:
            aabb = self.aabb
        x = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1]

        h_geo = 0
        h_tex = 0

        coords_list = [[0, 1], [0, 2], [1, 2]]
        
        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        for i in range(3):
            h_geo += self.sample_feature_plane2D(geo_feat_maps[i], x[..., coords_list[i]]) # (N, C)

        if self.use_tex:
            tex_feat_maps = [fm[:, self.geo_feat_dim:] for fm in feat_maps]
            tex_feat_maps = self.tex_convs(tex_feat_maps)
            for i in range(3):
                h_tex += self.sample_feature_plane2D(tex_feat_maps[i], x[..., coords_list[i]])

        h_geo = self.geo_decoder(h_geo) # (N, 1)
        if self.use_tex:
            h_tex = self.tex_decoder(h_tex).sigmoid() # (N, 1)
            h = torch.cat([h_geo, h_tex], dim=1)
        else:
            h = h_geo
        return h
    
    def forward(self, vol, x, aabb=None):
        feat_map = self.encode(vol)
        return self.decode(x, feat_map, aabb=aabb)


class AutoEncoderGroupSkip(nn.Module):
    def __init__(self, geo_feat_channels, tex_feat_channels, feat_channel_up, mlp_hidden_channels, mlp_hidden_layers, use_tex=True, tex_channels=3, posenc=0) -> None:
        super().__init__()
        self.use_tex = use_tex

        self.geo_encoder = nn.Conv3d(1, geo_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        if use_tex:
            self.tex_encoder = nn.Conv3d(tex_channels + 1, tex_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        out_channels = geo_feat_channels + tex_feat_channels if use_tex else geo_feat_channels
        self.norm = nn.InstanceNorm2d(out_channels)

        self.geo_feat_dim = geo_feat_channels
        self.tex_feat_dim = tex_feat_channels

        self.geo_convs = TriplaneGroupResnetBlock(
            geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
        )
        self.geo_decoder = DecoderMLPSkipConcat(feat_channel_up, 1, mlp_hidden_channels, mlp_hidden_layers)

        if use_tex:
            self.tex_convs = TriplaneGroupResnetBlock(
                tex_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
            )
            self.tex_decoder = DecoderMLPSkipConcat(feat_channel_up, tex_channels, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
        
        self.register_buffer("aabb", torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32))

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + list(self.tex_decoder.parameters())

    def reset_aabb(self, aabb):
        print("set net aabb:", aabb)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # self.register_buffer("aabb", aabb.to(self.encoder.weight.device))
        self.aabb = aabb.to(self.geo_encoder.weight.device)

    def encode(self, vol):
        geo_feat = self.geo_encoder(vol[:, :1])
        if self.use_tex:
            tex_feat = self.tex_encoder(vol)
            vol_feat = torch.cat([geo_feat, tex_feat], dim=1)
        else:
            vol_feat = geo_feat

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def decode(self, x, feat_maps, aabb=None):
        # x [N, 3]
        if aabb is None:
            aabb = self.aabb
        x = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1]

        h_geo = 0
        h_tex = 0

        coords_list = [[0, 1], [0, 2], [1, 2]]
        
        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        for i in range(3):
            h_geo += self.sample_feature_plane2D(geo_feat_maps[i], x[..., coords_list[i]]) # (N, C)

        if self.use_tex:
            tex_feat_maps = [fm[:, self.geo_feat_dim:] for fm in feat_maps]
            tex_feat_maps = self.tex_convs(tex_feat_maps)
            for i in range(3):
                h_tex += self.sample_feature_plane2D(tex_feat_maps[i], x[..., coords_list[i]])

        h_geo = self.geo_decoder(h_geo) # (N, 1)
        if self.use_tex:
            h_tex = self.tex_decoder(h_tex).sigmoid() # (N, 1)
            h = torch.cat([h_geo, h_tex], dim=1)
        else:
            h = h_geo
        return h
    
    def forward(self, vol, x, aabb=None):
        feat_map = self.encode(vol)
        return self.decode(x, feat_map, aabb=aabb)


class AutoEncoderGroupPBR(nn.Module):
    def __init__(self, geo_feat_channels, tex_feat_channels, feat_channel_up, mlp_hidden_channels, mlp_hidden_layers, use_tex=True, tex_channels=3, posenc=0) -> None:
        super().__init__()
        self.use_tex = use_tex

        self.geo_encoder = nn.Conv3d(1, geo_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        if use_tex:
            self.tex_encoder = nn.Conv3d(tex_channels + 1, tex_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        out_channels = geo_feat_channels + tex_feat_channels if use_tex else geo_feat_channels
        self.norm = nn.InstanceNorm2d(out_channels)

        self.geo_feat_dim = geo_feat_channels
        self.tex_feat_dim = tex_feat_channels

        self.geo_convs = TriplaneGroupResnetBlock(
            geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
        )
        self.geo_decoder = DecoderMLPSkipConcat(feat_channel_up, 1, mlp_hidden_channels, mlp_hidden_layers)

        if use_tex:
            self.tex_convs = nn.Sequential(
                TriplaneGroupResnetBlock(tex_feat_channels, feat_channel_up, ks=3, input_norm=False, input_act=False),
                TriplaneGroupResnetBlock(feat_channel_up, feat_channel_up, ks=3, input_norm=True, input_act=True),
            )
            self.rgb_decoder = DecoderMLPSkipConcat(feat_channel_up, 3, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
            self.mr_decoder = DecoderMLPSkipConcat(feat_channel_up, 2, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
            self.normal_decoder = DecoderMLPSkipConcat(feat_channel_up, 3, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
        
        self.register_buffer("aabb", torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32))

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + \
              list(self.rgb_decoder.parameters()) + list(self.mr_decoder.parameters()) + list(self.normal_decoder.parameters())

    def reset_aabb(self, aabb):
        print("set net aabb:", aabb)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # self.register_buffer("aabb", aabb.to(self.encoder.weight.device))
        self.aabb = aabb.to(self.geo_encoder.weight.device)

    def encode(self, vol):
        geo_feat = self.geo_encoder(vol[:, :1])
        if self.use_tex:
            tex_feat = self.tex_encoder(vol)
            vol_feat = torch.cat([geo_feat, tex_feat], dim=1)
        else:
            vol_feat = geo_feat

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def decode(self, x, feat_maps, aabb=None):
        # x [N, 3]
        if aabb is None:
            aabb = self.aabb
        x = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1]

        h_geo = 0
        h_tex = 0

        coords_list = [[0, 1], [0, 2], [1, 2]]
        
        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        for i in range(3):
            h_geo += self.sample_feature_plane2D(geo_feat_maps[i], x[..., coords_list[i]]) # (N, C)

        if self.use_tex:
            tex_feat_maps = [fm[:, self.geo_feat_dim:] for fm in feat_maps]
            tex_feat_maps = self.tex_convs(tex_feat_maps)
            for i in range(3):
                h_tex += self.sample_feature_plane2D(tex_feat_maps[i], x[..., coords_list[i]])

        h_geo = self.geo_decoder(h_geo) # (N, 1)
        if self.use_tex:
            h_rgb = self.rgb_decoder(h_tex) # (N, 3)
            h_mr = self.mr_decoder(h_tex) # (N, 2)
            h_normal = self.normal_decoder(h_tex) # (N, 3)
            h = torch.cat([h_geo, h_rgb, h_mr, h_normal], dim=1)
        else:
            h = h_geo
        return h
    
    def forward(self, vol, x, aabb=None):
        feat_map = self.encode(vol)
        return self.decode(x, feat_map, aabb=aabb)

