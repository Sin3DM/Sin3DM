"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1) # .cuda()
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1) # .cuda()

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        # if torch.cuda.is_available():
        #     state_dict = torch.load('metrics/lpips_weights.ckpt')
        # else:
        state_dict = torch.load('lpips_weights.ckpt',
                                map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
        y = (y - self.mu.to(x.device)) / self.sigma.to(x.device)
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(group_of_images, lpips_model=None):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    if lpips_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lpips = LPIPS().eval().to(device)
    else:
        lpips = lpips_model

    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.item()


@torch.no_grad()
def calculate_multiview_lpips_given_paths(gen_render_dirs, device="cuda:0"):
    from PIL import Image
    """Calculates multi view SIFID"""
    lpips_model = LPIPS().eval().to(device)

    n_views = len(os.listdir(gen_render_dirs[0]))
    print("lpips n_views:", n_views)
    
    lpips_list = []
    for i in tqdm(range(n_views), desc="Calculating multi-view LPIPS"):
        gen_render_paths = [os.path.join(gen_render_dir, f'{i:03d}.png') for gen_render_dir in gen_render_dirs]

        # read images and normalize to [-1, 1]
        # TODO: check if this is correct
        images = np.array([
            np.asarray(Image.open(fname).convert('RGB')) / 255. for fname in gen_render_paths
        ])
        images = (images[..., 0:3] - 0.5) / 0.5 # normalize to [-1, 1]
        images = images.transpose((0, 3, 1, 2))
        images = torch.from_numpy(images).type(torch.FloatTensor).to(device) # (N, C, H, W)

        lpips_view_value = calculate_lpips_given_images(images, lpips_model)
        lpips_list.append(lpips_view_value)
    lpips_avg = np.mean(lpips_list)
    return {"mv_lpips": lpips_avg}
