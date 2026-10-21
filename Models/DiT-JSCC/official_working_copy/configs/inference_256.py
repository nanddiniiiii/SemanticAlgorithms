"""Public 256 x 256 inference architecture."""

import ml_collections
import torch.nn as nn


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


class TransConfig:
    channel_number = [8, 8]
    channel_number_sem_list = list(range(4, 29, 2))
    channel_number_aux_list = list(range(0, 25, 2))
    chan_param = 0
    multiple_snr = 0
    channel_type = "awgn"
    model = "SwinJSCC_w/o_SAandRA"
    pass_channel = True
    downsample = 4
    image_dims = (3, 256, 256)
    encoder_kwargs = dict(
        img_size=(256, 256),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10],
        C=32,
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
        img_size=32,
        patch_size=2,
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
