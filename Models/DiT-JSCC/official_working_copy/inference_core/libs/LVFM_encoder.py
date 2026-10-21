import torch
import torch.nn.functional as F
from libs.visual_encoder import load_encoders
import numpy as np
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

class LVFM_encoder(torch.nn.Module):
    def __init__(
            self,
            visualenc_type,
            device="cuda",
            resolution=256,
            ):
        super(LVFM_encoder, self).__init__()
        self.encoder, self.encoder_type, self.architecture = load_encoders(
            visualenc_type, device, resolution)


    def forward(self, ori_images):
        #with torch.no_grad():
        raw_ori_image_ = self.preprocess_raw_image(ori_images, self.encoder_type)
        z_ori = self.encoder.forward_features(raw_ori_image_)
        if 'mocov3' in self.encoder_type:
            z_ori = z_ori = z_ori[:, 1:]
        if 'dinov2' in self.encoder_type:
            z_ori = z_ori['x_norm_patchtokens']
        return z_ori

    def preprocess_raw_image(self, x, enc_type):
        resolution = x.shape[-1]
        if 'clip' in enc_type:
            x = x / 255.
            x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
            x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
        elif 'mocov3' in enc_type or 'mae' in enc_type:
            x = x / 255.
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        elif 'dinov2' in enc_type:
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        elif 'dinov1' in enc_type:
            x = x / 255.
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        elif 'jepa' in enc_type:
            x = x / 255.
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

        return x
