"""Public 512 x 512 inference architecture."""

import ml_collections
import torch.nn as nn


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


class TransConfig:
    channel_number = [32, 32]
    channel_number_sem_list = list(range(16, 113, 8))
    channel_number_aux_list = list(range(0, 97, 8))
    chan_param = 0
    multiple_snr = 0
    channel_type = "awgn"
    model = "SwinJSCC_w/o_SAandRA"
    pass_channel = True
    downsample = 5
    image_dims = (3, 512, 512)
    encoder_kwargs = dict(
        img_size=(512, 512),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320, 320],
        depths=[2, 2, 6, 2, 2],
        num_heads=[4, 6, 8, 10, 10],
        C=128,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )


def get_config():
    config = ml_collections.ConfigDict()
    config.nnet = d(
        name="uvit",
        img_size=64,
        patch_size=4,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1001,
        use_checkpoint=False,
        condi=True,
        condi_layers=5,
        p_uncondi=0.1,
        trans_config=TransConfig(),
    )
    return config
