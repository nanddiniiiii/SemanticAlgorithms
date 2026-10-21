import math

from libs.swinjscc.decoder import *
from libs.swinjscc.encoder import *
from libs.swinjscc.channel import Channel
from random import choice
import torch.nn as nn

class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)

class SwinJSCC(nn.Module):
    def __init__(self, config, use_decoder=False):
        super(SwinJSCC, self).__init__()
        self.config = config
        self.use_decoder = use_decoder

        encoder_kwargs = config.encoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        if self.use_decoder:
            decoder_kwargs = config.decoder_kwargs
            self.decoder = create_decoder(**decoder_kwargs)

        self.channel = Channel(config)
        self.pass_channel = config.pass_channel
        self.H = self.W = 0
        self.multiple_snr = config.multiple_snr
        # self.channel_number = config.channel_number
        # for i in range(len(self.channel_number)):
        #     self.channel_number[i] = int(self.channel_number[i])
        self.downsample = config.downsample
        self.model = config.model

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature, h = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature, h

    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            if self.use_decoder:
                self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W


        if given_SNR is not None:
            chan_param = given_SNR
        elif isinstance(self.config.multiple_snr, list) and self.training:
            chan_param = choice(self.multiple_snr)
        else:
            chan_param = self.config.chan_param

        feature = self.encoder(input_image, chan_param, self.config.channel_number, self.model)
        CBR = feature.numel() / 2 / input_image.numel()
        if self.pass_channel:
            noisy_feature, h = self.feature_pass_channel(feature, self.config.chan_param)
        else:
            noisy_feature = feature
            h=None
        if not self.use_decoder:
            return noisy_feature, feature, h
        else:
            recon_image = self.decoder(noisy_feature, self.config.chan_param, self.model)
            return noisy_feature, feature, recon_image

def get_trans_net(config, use_decoder=False):
    if config.net_type == "swinjscc":
        return SwinJSCC(config, use_decoder)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=768, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

from libs.transformer import Block, InterBlock
import torch

class SemanticJSCC(nn.Module):
    def __init__(self, config):
        super(SemanticJSCC, self).__init__()
        self.config = config
        self.pos_embed4enc = nn.Parameter(torch.zeros(1, 256, 256))
        self.patch_embed4enc = PatchEmbed(patch_size=1, in_chans=768, embed_dim=256)
        self.encoder = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])
        self.linear4enc = nn.Linear(256, 2)
        self.norm4enc = nn.LayerNorm(256)

        self.linear4dec = nn.Linear(2, 256)
        self.decoder = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])


        self.channel = Channel(config)
        self.pass_channel = config.pass_channel
        self.H = self.W = 0
        self.multiple_snr = config.multiple_snr


    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature, h = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature, h

    def forward(self, y_sem, given_SNR=None):
        B, L, C = y_sem.shape
        if given_SNR is not None:
            chan_param = given_SNR
        elif isinstance(self.config.multiple_snr, list) and self.training:
            chan_param = choice(self.multiple_snr)
        else:
            chan_param = self.config.chan_param
        y_sem = self.patch_embed4enc(y_sem) + self.pos_embed4enc
        for blk in self.encoder:
            y_sem = blk(y_sem)
        y_sem = self.norm4enc(y_sem)
        feature = self.linear4enc(y_sem)
        if self.pass_channel:
            noisy_feature, h = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
            h = None

        x = self.linear4dec(noisy_feature)
        for blk in self.decoder:
            x = blk(x)
        recon_y_sem = x
        return recon_y_sem

class InterJSCC(nn.Module):
    def __init__(self, config):
        super(InterJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        self.encoder = create_encoder_v1(**encoder_kwargs)
        self.linear = nn.Linear(config.encoder_kwargs["embed_dims"][-1], config.encoder_kwargs["C"])
        self.linear_inverse = nn.Linear(config.encoder_kwargs["C"], config.encoder_kwargs["embed_dims"][-1])
        self.pos_embed4enc = nn.Parameter(torch.zeros(1, 256, 256))
        self.patch_embed4enc = PatchEmbed(patch_size=1, in_chans=768, embed_dim=256)
        self.encoder_sem = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])

        self.encoder_aux = nn.ModuleList([
            Block(
                dim=320, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])

        self.linear4enc = nn.Linear(256, 2)
        self.norm4enc = nn.LayerNorm(256)

        self.linear4dec = nn.Linear(2, 256)
        self.decoder_sem = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])

        # self.inter_blocks_4enc = nn.ModuleList([
        #     InterBlock(
        #         dim1=320, dim2=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
        #         act_layer=nn.GELU, use_checkpoint=True)
        #     for i in range(4)])

        # self.inter_blocks_4dec = nn.ModuleList([
        #     InterBlock(
        #         dim1=320, dim2=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
        #         act_layer=nn.GELU, use_checkpoint=True)
        #     for i in range(4)])

        self.channel = Channel(config)
        self.pass_channel = config.pass_channel
        self.H = self.W = 0
        self.multiple_snr = config.multiple_snr
        self.model = config.model

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature, h = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature, h

    def forward(self, input_image, y_sem, given_SNR=None):

        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.H = H
            self.W = W


        if given_SNR is not None:
            chan_param = given_SNR
        elif isinstance(self.config.multiple_snr, list) and self.training:
            chan_param = choice(self.multiple_snr)
        else:
            chan_param = self.config.chan_param

        feature = self.encoder(input_image, chan_param, self.config.channel_number, self.model)

        y_sem = self.patch_embed4enc(y_sem) + self.pos_embed4enc

        for blk, blk_aux in zip(self.encoder_sem, self.encoder_aux):
            feature = blk_aux(feature)
            y_sem = blk(y_sem)

        # for inter_blk in self.inter_blocks_4enc:
        #     feature, y_sem = inter_blk(feature, y_sem)

        # for blk in self.encoder_sem:
        #     y_sem = blk(y_sem)

        y_sem = self.norm4enc(y_sem)
        y_sem= self.linear4enc(y_sem)
        feature = self.linear(feature)

        noisy_y_sem, h = self.feature_pass_channel(y_sem, chan_param)
        noisy_feature, h = self.feature_pass_channel(feature, chan_param)
        z = self.linear_inverse(noisy_feature)
        x = self.linear4dec(noisy_y_sem)
        # for blk, inter_blk in zip(self.decoder_sem, self.inter_blocks_4dec):
        #     x = blk(x)
        #     z, x = inter_blk(z, x)
        for blk in self.decoder_sem:
            x = blk(x)
        recon_y_sem = x
        noisy_feature = z
        return noisy_feature, recon_y_sem

# class InterJSCC(nn.Module):
#     def __init__(self, config):
#         super(InterJSCC, self).__init__()
#         self.config = config
#         encoder_kwargs = config.encoder_kwargs
#         self.pos_embed4enc = nn.Parameter(torch.zeros(1, 256, 256))
#         self.patch_embed4enc = PatchEmbed(patch_size=1, in_chans=768, embed_dim=256)
#         self.encoder_sem = nn.ModuleList([
#             Block(
#                 dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
#                 norm_layer=nn.LayerNorm, use_checkpoint=True)
#             for i in range(4)])
#
#
#
#         self.linear4enc = nn.Linear(256, 16)
#         self.norm4enc = nn.LayerNorm(256)
#
#         self.linear4dec = nn.Linear(16, 256)
#         self.decoder_sem = nn.ModuleList([
#             Block(
#                 dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
#                 norm_layer=nn.LayerNorm, use_checkpoint=True)
#             for i in range(4)])
#
#         self.channel = Channel(config)
#         self.pass_channel = config.pass_channel
#         self.H = self.W = 0
#         self.multiple_snr = config.multiple_snr
#         self.model = config.model
#
#     def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
#         noisy_feature, h = self.channel.forward(feature, chan_param, avg_pwr)
#         return noisy_feature, h
#
#     def forward(self, input_image, y_sem, given_SNR=None):
#
#         B, _, H, W = input_image.shape
#
#         # if H != self.H or W != self.W:
#         #     self.encoder.update_resolution(H, W)
#         #     self.H = H
#         #     self.W = W
#
#
#         if given_SNR is not None:
#             chan_param = given_SNR
#         elif isinstance(self.config.multiple_snr, list) and self.training:
#             chan_param = choice(self.multiple_snr)
#         else:
#             chan_param = self.config.chan_param
#
#         y_sem = self.patch_embed4enc(y_sem) + self.pos_embed4enc
#
#         for blk in self.encoder_sem:
#             y_sem = blk(y_sem)
#
#
#         y_sem = self.norm4enc(y_sem)
#         y_sem= self.linear4enc(y_sem)
#
#         noisy_y_sem, h = self.feature_pass_channel(y_sem, chan_param)
#
#         x = self.linear4dec(noisy_y_sem)
#
#         for blk in self.decoder_sem:
#             x = blk(x)
#         recon_y_sem = x
#
#         return 0, recon_y_sem

class JSCC_2branch(nn.Module):
    def __init__(self, config):
        super(JSCC_2branch, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        self.encoder = create_encoder_v2(**encoder_kwargs)
        self.pos_embed4enc = nn.Parameter(torch.zeros(1, 256, 256))
        self.patch_embed4enc = PatchEmbed(patch_size=1, in_chans=768, embed_dim=256)
        self.encoder_sem = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])

        self.linear4enc = nn.Linear(256, 32)
        self.norm4enc = nn.LayerNorm(256)

        self.linear4dec = nn.Linear(32, 256)
        self.decoder_sem = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])

        self.hidden_dim = int(32 * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(32, self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = 32
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        self.channel = Channel(config)
        self.pass_channel = config.pass_channel
        self.H = self.W = 0
        self.multiple_snr = config.multiple_snr
        self.model = config.model

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature, h = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature, h

    def forward(self, input_image, y_sem, given_rate=None, given_snr=None):

        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.H = H
            self.W = W

        if given_rate is None and self.training:
            channel_number_s = choice(self.config.channel_number_sem_list)
            channel_number_a = choice(self.config.channel_number_aux_list)
        elif given_rate is None and (not self.training):
            channel_number_s = self.config.channel_number[0]
            channel_number_a = self.config.channel_number[1]
        else:
            channel_number_s = given_rate[0]
            channel_number_a = given_rate[1]

        chan_param = self.config.chan_param if given_snr is None else given_snr

        feature, mask_a, x_l4 = self.encoder(input_image, chan_param, channel_number_a, self.model)
        if mask_a.sum()==0:
            noisy_feature = feature
        else:
            avg_pwr = torch.sum(feature ** 2) / mask_a.sum()
            noisy_feature, h = self.feature_pass_channel(feature, chan_param, avg_pwr)
            noisy_feature = noisy_feature * mask_a
        y_sem = self.patch_embed4enc(y_sem) + self.pos_embed4enc

        for blk in self.encoder_sem:
            y_sem = blk(y_sem)
        y_sem = self.norm4enc(y_sem)
        y_sem = self.linear4enc(y_sem)
        rate_cuda = torch.tensor(channel_number_s, dtype=torch.float).to(y_sem.device)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](y_sem.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, y_sem.shape[1], -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        y_sem = y_sem * mod_val
        mask = torch.sum(mod_val, dim=1)
        sorted, indices = mask.sort(dim=1, descending=True)
        if channel_number_s == 0:
            mask = torch.zeros_like(y_sem)
            y_sem = y_sem * mask
            noisy_y_sem = y_sem
        else:
            c_indices = indices[:, :channel_number_s]
            add = torch.arange(
                0, B * y_sem.size(2), y_sem.size(2), device=y_sem.device
            ).unsqueeze(1).repeat(1, channel_number_s)
            c_indices = c_indices + add.int()
            mask = torch.zeros(mask.size(), device=y_sem.device).reshape(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, y_sem.size()[2])
            mask = mask.unsqueeze(1).expand(-1, y_sem.shape[1], -1)
            y_sem = y_sem * mask
            mask_s = mask
            avg_pwr = torch.sum(y_sem ** 2) / mask_s.sum()
            noisy_y_sem, h = self.feature_pass_channel(y_sem, chan_param, avg_pwr)
            noisy_y_sem = noisy_y_sem * mask_s
        x = self.linear4dec(noisy_y_sem)

        for blk in self.decoder_sem:
            x = blk(x)
        recon_y_sem = x

        return noisy_feature, recon_y_sem


class JSCC_2branch_512(nn.Module):
    def __init__(self, config):
        super(JSCC_2branch_512, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        self.encoder = create_encoder_v2(**encoder_kwargs)
        self.pos_embed4enc = nn.Parameter(torch.zeros(1, 256, 256))
        self.patch_embed4enc = PatchEmbed(patch_size=2, in_chans=768, embed_dim=256)
        self.encoder_sem = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])
        self.bottleneck = 128
        self.linear4enc = nn.Linear(256, self.bottleneck)
        self.norm4enc = nn.LayerNorm(256)

        self.linear4dec = nn.Linear(self.bottleneck, 256)
        self.decoder_sem = nn.ModuleList([
            Block(
                dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                norm_layer=nn.LayerNorm, use_checkpoint=True)
            for i in range(4)])

        self.hidden_dim = int(self.bottleneck * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.bottleneck, self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.bottleneck
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        self.channel = Channel(config)
        self.pass_channel = config.pass_channel
        self.H = self.W = 0
        self.multiple_snr = config.multiple_snr
        self.model = config.model

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature, h = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature, h

    def forward(self, input_image, y_sem, given_rate=None, given_snr=None):

        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.H = H
            self.W = W

        if given_rate is None and self.training:
            channel_number_s = choice(self.config.channel_number_sem_list)
            channel_number_a = choice(self.config.channel_number_aux_list)
        elif given_rate is None and (not self.training):
            channel_number_s = self.config.channel_number[0]
            channel_number_a = self.config.channel_number[1]
        else:
            channel_number_s = given_rate[0]
            channel_number_a = given_rate[1]

        chan_param = self.config.chan_param if given_snr is None else given_snr

        feature, mask_a, x_l4 = self.encoder(input_image, chan_param, channel_number_a, self.model)
        if mask_a.sum()==0:
            noisy_feature = feature
        else:
            avg_pwr = torch.sum(feature ** 2) / mask_a.sum()
            noisy_feature, h = self.feature_pass_channel(feature, chan_param, avg_pwr)
            noisy_feature = noisy_feature * mask_a
        y_sem = self.patch_embed4enc(y_sem) + self.pos_embed4enc

        for blk in self.encoder_sem:
            y_sem = blk(y_sem)
        y_sem = self.norm4enc(y_sem)
        y_sem = self.linear4enc(y_sem)
        rate_cuda = torch.tensor(channel_number_s, dtype=torch.float).to(y_sem.device)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](y_sem.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, y_sem.shape[1], -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        y_sem = y_sem * mod_val
        mask = torch.sum(mod_val, dim=1)
        sorted, indices = mask.sort(dim=1, descending=True)
        if channel_number_s == 0:
            mask = torch.zeros_like(y_sem)
            y_sem = y_sem * mask
            noisy_y_sem = y_sem
        else:
            c_indices = indices[:, :channel_number_s]
            add = torch.arange(
                0, B * y_sem.size(2), y_sem.size(2), device=y_sem.device
            ).unsqueeze(1).repeat(1, channel_number_s)
            c_indices = c_indices + add.int()
            mask = torch.zeros(mask.size(), device=y_sem.device).reshape(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, y_sem.size()[2])
            mask = mask.unsqueeze(1).expand(-1, y_sem.shape[1], -1)
            y_sem = y_sem * mask
            mask_s = mask
            avg_pwr = torch.sum(y_sem ** 2) / mask_s.sum()
            noisy_y_sem, h = self.feature_pass_channel(y_sem, chan_param, avg_pwr)
            noisy_y_sem = noisy_y_sem * mask_s
        x = self.linear4dec(noisy_y_sem)

        for blk in self.decoder_sem:
            x = blk(x)
        recon_y_sem = x

        return noisy_feature, recon_y_sem
