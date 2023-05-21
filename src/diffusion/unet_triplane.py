from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    SiLU,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from utils.triplane_util import compose_featmaps, decompose_featmaps


class TriplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, is_rollout=True) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout

        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        if self.is_rollout:
            tpl_xy_h = th.cat([tpl_xy,
                            th.mean(tpl_yz, dim=-1, keepdim=True).transpose(-1, -2).expand_as(tpl_xy),
                            th.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy)], dim=1) # [B, C * 3, H, W]
            tpl_xz_h = th.cat([tpl_xz,
                                th.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                                th.mean(tpl_yz, dim=-2, keepdim=True).expand_as(tpl_xz)], dim=1) # [B, C * 3, H, D]
            tpl_yz_h = th.cat([tpl_yz,
                            th.mean(tpl_xy, dim=-2, keepdim=True).transpose(-1, -2).expand_as(tpl_yz),
                            th.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_yz)], dim=1) # [B, C * 3, W, D]
        else:
            tpl_xy_h = tpl_xy
            tpl_xz_h = tpl_xz
            tpl_yz_h = tpl_yz
        
        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        tpl_xy_h = self.conv_xy(tpl_xy_h)
        tpl_xz_h = self.conv_xz(tpl_xz_h)
        tpl_yz_h = self.conv_yz(tpl_yz_h)

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm_xy = normalization(channels)
        self.norm_xz = normalization(channels)
        self.norm_yz = normalization(channels)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy_h = self.norm_xy(tpl_xy) # [B, C, H, W]
        tpl_xz_h = self.norm_xz(tpl_xz) # [B, C, H, D]
        tpl_yz_h = self.norm_yz(tpl_yz) # [B, C, W, D]

        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)
    

class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        return (self.silu(tpl_xy), self.silu(tpl_xz), self.silu(tpl_yz))


# class TriplaneIdentity(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, featmaps):
#         return featmaps


class TriplaneUpsample2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='bilinear', align_corners=False)
        tpl_xz = F.interpolate(tpl_xz, scale_factor=2, mode='bilinear', align_corners=False)
        tpl_yz = F.interpolate(tpl_yz, scale_factor=2, mode='bilinear', align_corners=False)

        assert tpl_xy.shape[-2] == H * 2 and tpl_xy.shape[-1] == W * 2
        assert tpl_xz.shape[-2] == H * 2 and tpl_xz.shape[-1] == D * 2
        assert tpl_yz.shape[-2] == W * 2 and tpl_yz.shape[-1] == D * 2

        return (tpl_xy, tpl_xz, tpl_yz)


class TriplaneDownsample2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy = F.avg_pool2d(tpl_xy, kernel_size=2, stride=2)
        tpl_xz = F.avg_pool2d(tpl_xz, kernel_size=2, stride=2)
        tpl_yz = F.avg_pool2d(tpl_yz, kernel_size=2, stride=2)

        assert tpl_xy.shape[-2] == H // 2 and tpl_xy.shape[-1] == W // 2
        assert tpl_xz.shape[-2] == H // 2 and tpl_xz.shape[-1] == D // 2
        assert tpl_yz.shape[-2] == W // 2 and tpl_yz.shape[-1] == D // 2

        return (tpl_xy, tpl_xz, tpl_yz)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TriplaneResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        is_rollout=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            TriplaneNorm(channels),
            TriplaneSiLU(),
            TriplaneConv(channels, self.out_channels, 3, padding=1, is_rollout=is_rollout),
        )

        self.updown = up or down

        if up:
            self.h_upd = TriplaneUpsample2x()
            self.x_upd = TriplaneUpsample2x()
        elif down:
            self.h_upd = TriplaneDownsample2x()
            self.x_upd = TriplaneDownsample2x()
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            TriplaneNorm(self.out_channels),
            TriplaneSiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                TriplaneConv(self.out_channels, self.out_channels, 3, padding=1, is_rollout=is_rollout)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = TriplaneConv(
                channels, self.out_channels, 3, padding=1, is_rollout=False
            )
        else:
            self.skip_connection = TriplaneConv(channels, self.out_channels, 1, padding=0, is_rollout=False)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # x: (h_xy, h_xz, h_yz)
        if self.updown:
            raise NotImplementedError
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x, H, W, D)
            h = self.h_upd(h, H, W, D)
            x = self.x_upd(x, H, W, D)
            h = in_conv(h, H, W, D)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h[0].dtype)
        while len(emb_out.shape) < len(h[0].shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            h = out_norm(h)
            h_xy, h_xz, h_yz = h
            h_xy = h_xy * (1 + scale) + shift
            h_xz = h_xz * (1 + scale) + shift
            h_yz = h_yz * (1 + scale) + shift
            h = (h_xy, h_xz, h_yz)
            # h = out_norm(h) * (1 + scale) + shift

            h = out_rest(h)
        else:
            h_xy, h_xz, h_yz = h
            h_xy = h_xy + emb_out
            h_xz = h_xz + emb_out
            h_yz = h_yz + emb_out
            h = (h_xy, h_xz, h_yz)
            # h = h + emb_out

            h = self.out_layers(h)
        
        x_skip = self.skip_connection(x)
        x_skip_xy, x_skip_xz, x_skip_yz = x_skip
        h_xy, h_xz, h_yz = h
        return (h_xy + x_skip_xy, h_xz + x_skip_xz, h_yz + x_skip_yz)
        # return self.skip_connection(x) + h


class TriplaneUNetModelSmall(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks=1,
        dropout=0,
        channel_mult=(1, 2),
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        dims = 2
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.in_conv = TimestepEmbedSequential(TriplaneConv(in_channels, ch, 1, padding=0, is_rollout=False))
        print("In conv: TriplaneConv")

        input_block_chans = [ch]
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            layers = []
            if level != 0:
                layers.append(TriplaneDownsample2x())
                print(f"Down level {level}: TriplaneDownsample2x, ch {ch}")

            for _ in range(num_res_blocks):
                layers.append(
                    TriplaneResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                print(f"Down level {level} block {_}: TriplaneResBlock, ch {ch}")

            ch = int(mult * model_channels)
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans.append(ch)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers.append(
                    TriplaneResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                print(f"Up level {level} block {i}: TriplaneResBlock, ch {ch}")

            ch = int(model_channels * mult)
            if level > 0:
                layers.append(
                    TriplaneUpsample2x()
                )
                print(f"Up level {level}: TriplaneUpsample2x, ch {ch}")

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        # self.out = nn.Sequential(
        #     normalization(ch),
        #     SiLU(),
        #     zero_module(conv_nd(dims, input_ch, out_channels, 1, padding=0)),
        # )
        self.out = nn.Sequential(
            TriplaneNorm(ch),
            TriplaneSiLU(),
            zero_module(TriplaneConv(input_ch, out_channels, 1, padding=0, is_rollout=False))
        )
        print("Out conv: TriplaneConv")

        print(f"number of input blocks: {len(self.input_blocks)}")
        print(f"number of output blocks: {len(self.output_blocks) + 1}")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, H=None, W=None, D=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert H is not None and W is not None and D is not None

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype) if y is None else th.cat([x, y], dim=1).type(self.dtype)
        h_triplane = decompose_featmaps(h, (H, W, D))

        h_triplane = self.in_conv(h_triplane, emb)

        for level, module in enumerate(self.input_blocks):
            h_triplane = module(h_triplane, emb)
            hs.append(h_triplane)

        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h_triplane = hs.pop()
            else:
                h_triplane_pop = hs.pop()
                h_triplane = (th.cat([h_triplane[0], h_triplane_pop[0]], dim=1),
                              th.cat([h_triplane[1], h_triplane_pop[1]], dim=1),
                              th.cat([h_triplane[2], h_triplane_pop[2]], dim=1))
            
            h_triplane = module(h_triplane, emb)
        
        h_triplane = self.out(h_triplane)
        h = compose_featmaps(*h_triplane)[0]
        assert h.shape == x.shape
        return h


class TriplaneUNetModelSmallRaw(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks=1,
        dropout=0,
        channel_mult=(1, 2),
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        dims = 2
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.in_conv = TimestepEmbedSequential(TriplaneConv(in_channels, ch, 1, padding=0, is_rollout=False))
        print("In conv: TriplaneConv")

        input_block_chans = [ch]
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            layers = []
            if level != 0:
                layers.append(TriplaneDownsample2x())
                print(f"Down level {level}: TriplaneDownsample2x, ch {ch}")

            for _ in range(num_res_blocks):
                layers.append(
                    TriplaneResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        is_rollout=False,
                    )
                )
                print(f"Down level {level} block {_}: TriplaneResBlock, ch {ch}")

            ch = int(mult * model_channels)
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans.append(ch)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers.append(
                    TriplaneResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        is_rollout=False,
                    )
                )
                print(f"Up level {level} block {i}: TriplaneResBlock, ch {ch}")

            ch = int(model_channels * mult)
            if level > 0:
                layers.append(
                    TriplaneUpsample2x()
                )
                print(f"Up level {level}: TriplaneUpsample2x, ch {ch}")

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        # self.out = nn.Sequential(
        #     normalization(ch),
        #     SiLU(),
        #     zero_module(conv_nd(dims, input_ch, out_channels, 1, padding=0)),
        # )
        self.out = nn.Sequential(
            TriplaneNorm(ch),
            TriplaneSiLU(),
            zero_module(TriplaneConv(input_ch, out_channels, 1, padding=0, is_rollout=False))
        )
        print("Out conv: TriplaneConv")

        print(f"number of input blocks: {len(self.input_blocks)}")
        print(f"number of output blocks: {len(self.output_blocks) + 1}")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, H=None, W=None, D=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert H is not None and W is not None and D is not None

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype) if y is None else th.cat([x, y], dim=1).type(self.dtype)
        h_triplane = decompose_featmaps(h, (H, W, D))

        h_triplane = self.in_conv(h_triplane, emb)

        for level, module in enumerate(self.input_blocks):
            h_triplane = module(h_triplane, emb)
            hs.append(h_triplane)

        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h_triplane = hs.pop()
            else:
                h_triplane_pop = hs.pop()
                h_triplane = (th.cat([h_triplane[0], h_triplane_pop[0]], dim=1),
                              th.cat([h_triplane[1], h_triplane_pop[1]], dim=1),
                              th.cat([h_triplane[2], h_triplane_pop[2]], dim=1))
            
            h_triplane = module(h_triplane, emb)
        
        h_triplane = self.out(h_triplane)
        h = compose_featmaps(*h_triplane)[0]
        assert h.shape == x.shape
        return h
