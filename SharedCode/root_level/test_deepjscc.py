import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append('Deep-JSCC-PyTorch')
from channel import Channel
from model import _ConvWithPReLU, _TransConvWithPReLU

# Exact Architecture matching the checkpoint
class _Encoder_Exact(nn.Module):
    def __init__(self, P=1):
        super().__init__()
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=19, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            batch_size = z_hat.size()[0] if z_hat.dim() == 4 else 1
            k = torch.prod(torch.tensor(z_hat.size()[1:])) if z_hat.dim() == 4 else torch.prod(torch.tensor(z_hat.size()))
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            return tensor.squeeze(0) if batch_size == 1 else tensor
        return _inner

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.norm(x)
        return x

class _Decoder_Exact(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = _TransConvWithPReLU(in_channels=19, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=6, stride=2, padding=2, output_padding=0)
        self.tconv5 = _TransConvWithPReLU(in_channels=16, out_channels=3, kernel_size=6, stride=2, padding=2, output_padding=0, activate=nn.Sigmoid())

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x

class DeepJSCC_Exact(nn.Module):
    def __init__(self, channel_type='AWGN', snr=10):
        super().__init__()
        self.encoder = _Encoder_Exact()
        self.channel = Channel(channel_type, snr)
        self.decoder = _Decoder_Exact()

    def forward(self, x):
        z = self.encoder(x)
        z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

# Load new_img
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
targets = np.array(dataset.targets)
new_index = int(np.where(targets == 1)[0][5])
new_img = torchvision.transforms.functional.to_pil_image(dataset.data[new_index])

test_img_tensor = T.ToTensor()(new_img).unsqueeze(0)

# Load Model
model = DeepJSCC_Exact(channel_type='AWGN', snr=10)
chk_path = 'Deep-JSCC-PyTorch/out/imagenet_10_0.33_200.00_32_19.pth'
state_dict = torch.load(chk_path, map_location='cpu')
model.load_state_dict(state_dict, strict=True)
model.eval()

with torch.no_grad():
    latent = model.encoder(test_img_tensor)
    # Simulate channel transmission
    received_latent = model.channel(latent)
    reconstructed_tensor = model.decoder(received_latent)

recon_img = T.ToPILImage()(reconstructed_tensor.squeeze(0).clamp(0, 1))

# Metrics
orig_np = np.array(new_img)
recon_np = np.array(recon_img)
val_psnr = psnr(orig_np, recon_np, data_range=255)

symbols = int(np.prod(latent.shape[1:]))
orig_bits = 32 * 32 * 3
cbr = symbols / orig_bits

print(f"Latent Shape: {latent.shape}")
print(f"CBR: {cbr:.4f} ({symbols} symbols)")
print(f"PSNR: {val_psnr:.2f} dB")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(new_img)
axes[0].set_title("ORIGINAL IMAGE")
axes[0].axis('off')

axes[1].imshow(recon_img)
axes[1].set_title(f"RECONSTRUCTED (DeepJSCC Exact)\\nPSNR: {val_psnr:.2f} dB")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("deepjscc_test.png")
print("Saved visualization to deepjscc_test.png")
