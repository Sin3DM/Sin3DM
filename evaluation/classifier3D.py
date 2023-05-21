# from https://github.com/czq142857/DECOR-GAN/blob/main/evalFID.py
import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):
    def __init__(self, ef_dim=32, z_dim=512, class_num=24, voxel_size=128):
        super(classifier, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.class_num = class_num
        self.voxel_size = voxel_size

        self.conv_1 = nn.Conv3d(1,             self.ef_dim,   4, stride=2, padding=1, bias=True)
        self.bn_1 = nn.InstanceNorm3d(self.ef_dim)

        self.conv_2 = nn.Conv3d(self.ef_dim,   self.ef_dim*2, 4, stride=2, padding=1, bias=True)
        self.bn_2 = nn.InstanceNorm3d(self.ef_dim*2)

        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
        self.bn_3 = nn.InstanceNorm3d(self.ef_dim*4)

        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
        self.bn_4 = nn.InstanceNorm3d(self.ef_dim*8)

        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=2, padding=1, bias=True)

        if self.voxel_size==256:
            self.bn_5 = nn.InstanceNorm3d(self.z_dim)
            self.conv_5_2 = nn.Conv3d(self.z_dim, self.z_dim, 4, stride=2, padding=1, bias=True)

        self.linear1 = nn.Linear(self.z_dim, self.class_num, bias=True)

    def forward(self, inputs, out_layer=None, is_training=False):
        out = inputs

        out = self.bn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if out_layer == 1:
            return out

        out = self.bn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if out_layer == 2:
            return out

        out = self.bn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if out_layer == 3:
            return out

        out = self.bn_4(self.conv_4(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if out_layer == 4:
            return out

        if self.voxel_size==256:
            out = self.bn_5(out)
            out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
            out = self.conv_5_2(out)

        z = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
        z = z.view(-1,self.z_dim)
        out = F.leaky_relu(z, negative_slope=0.01, inplace=True)
        
        out = self.linear1(out)

        return out, z