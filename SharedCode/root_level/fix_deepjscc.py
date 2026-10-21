import json
import os

notebook_path = r"d:\<local OneDrive>\Desktop\image_semcom\ImageFeatures (1).ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    updated = False
    
    # We will look for the DeepJSCC cell (Cell 2) which has model_name = "DeepJSCC" and sys.path.insert
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            if 'model_name = "DeepJSCC"' in source and 'sys.path.insert' in source:
                new_source = [
                    "model_name = \"DeepJSCC\"\n",
                    "print(f\"--- Evaluating {model_name} ---\")\n",
                    "\n",
                    "try:\n",
                    "    sys.path.insert(0, os.path.abspath('Deep-JSCC-PyTorch'))\n",
                    "    from model import _ConvWithPReLU, _TransConvWithPReLU, _Encoder, _Decoder\n",
                    "    import torch.nn as nn\n",
                    "    import torch\n",
                    "    \n",
                    "    # Custom architecture to match checkpoint exactly\n",
                    "    class EncoderCustom(nn.Module):\n",
                    "        def __init__(self):\n",
                    "            super().__init__()\n",
                    "            self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)\n",
                    "            self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)\n",
                    "            self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)\n",
                    "            self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)\n",
                    "            self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=19, kernel_size=5, padding=2)\n",
                    "            self.norm = _Encoder._normlizationLayer()\n",
                    "        def forward(self, x):\n",
                    "            x = self.conv1(x)\n",
                    "            x = self.conv2(x)\n",
                    "            x = self.conv3(x)\n",
                    "            x = self.conv4(x)\n",
                    "            x = self.conv5(x)\n",
                    "            x = self.norm(x)\n",
                    "            return x\n",
                    "\n",
                    "    class DecoderCustom(nn.Module):\n",
                    "        def __init__(self):\n",
                    "            super().__init__()\n",
                    "            self.tconv1 = _TransConvWithPReLU(in_channels=19, out_channels=32, kernel_size=5, stride=1, padding=2)\n",
                    "            self.tconv2 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)\n",
                    "            self.tconv3 = _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)\n",
                    "            self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=6, stride=2, padding=2, output_padding=1)\n",
                    "            self.tconv5 = _TransConvWithPReLU(in_channels=16, out_channels=3, kernel_size=6, stride=2, padding=2, output_padding=1, activate=nn.Sigmoid())\n",
                    "        def forward(self, x):\n",
                    "            x = self.tconv1(x)\n",
                    "            x = self.tconv2(x)\n",
                    "            x = self.tconv3(x)\n",
                    "            x = self.tconv4(x)\n",
                    "            x = self.tconv5(x)\n",
                    "            return x\n",
                    "\n",
                    "    class DeepJSCC_Custom(nn.Module):\n",
                    "        def __init__(self):\n",
                    "            super().__init__()\n",
                    "            self.encoder = EncoderCustom()\n",
                    "            self.decoder = DecoderCustom()\n",
                    "        def forward(self, x):\n",
                    "            z = self.encoder(x)\n",
                    "            return self.decoder(z)\n",
                    "\n",
                    "    chk_path = 'Deep-JSCC-PyTorch/out/imagenet_10_0.33_200.00_32_19.pth'\n",
                    "    \n",
                    "    model_dj = DeepJSCC_Custom()\n",
                    "    \n",
                    "    if os.path.exists(chk_path):\n",
                    "        state_dict = torch.load(chk_path, map_location='cpu')\n",
                    "        model_dj.load_state_dict(state_dict, strict=True)\n",
                    "    else:\n",
                    "        print(f\"Checkpoint not found at {chk_path}\")\n",
                    "    \n",
                    "    if torch.cuda.is_available():\n",
                    "        model_dj = model_dj.cuda()\n",
                    "        \n",
                    "    model_dj.eval()\n",
                    "    \n",
                    "    with torch.no_grad():\n",
                    "        latent = model_dj.encoder(test_img_tensor)\n",
                    "        reconstructed_tensor = model_dj.decoder(latent)\n",
                    "        \n",
                    "    latent_shape = str(tuple(latent.shape))\n",
                    "    latent_type = \"Continuous Channel Symbols\"\n",
                    "    \n",
                    "    symbols = int(np.prod(latent.shape[1:]))\n",
                    "    cbr = symbols / (32*32*3)\n",
                    "    tx_size = f\"{symbols} symbols (CBR={cbr:.3f})\"\n",
                    "    \n",
                    "    print(f\"Latent Shape: {latent_shape}\")\n",
                    "    print(f\"Latent Type: {latent_type}\")\n",
                    "    \n",
                    "    recon_img = T.ToPILImage()(reconstructed_tensor.squeeze(0).cpu().clamp(0, 1))\n",
                    "    plot_reconstruction(test_img, recon_img, model_name)\n",
                    "    \n",
                    "    val_psnr, val_ssim, val_lpips = calculate_metrics(test_img, recon_img)\n",
                    "    \n",
                    "    # Remove old entry if this cell was run before\n",
                    "    all_results = [r for r in all_results if r['Model'] != model_name]\n",
                    "    \n",
                    "    all_results.append({\n",
                    "        \"Model\": model_name, \"Pretrained\": \"Yes (imagenet_10_0.33)\", \"Latent Shape\": latent_shape, \n",
                    "        \"Original Image Bits\": ORIG_BITS,\n",
                    "        \"Transmission Size / Rate\": tx_size, \"Compression Ratio\": \"N/A (Analog)\", \"Bit Reduction %\": \"N/A\",\n",
                    "        \"Reconstruction Level / Operating Point\": f\"Fixed SNR=10, CBR={cbr:.3f}\", \"PSNR\": f\"{val_psnr:.2f}\" if val_psnr else \"N/A\", \n",
                    "        \"SSIM\": f\"{val_ssim:.4f}\" if val_ssim else \"N/A\", \"LPIPS\": f\"{val_lpips:.4f}\" if val_lpips else \"N/A\",\n",
                    "        \"recon_img\": recon_img \n",
                    "    })\n",
                    "    print(f\"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}\")\n",
                    "    sys.path.pop(0)\n",
                    "\n",
                    "except Exception as e:\n",
                    "    import traceback\n",
                    "    print(f\"{model_name} evaluation failed: {e}\")\n",
                    "    traceback.print_exc()\n",
                    "    sys.path.pop(0) if sys.path[0] == os.path.abspath('Deep-JSCC-PyTorch') else None\n"
                ]
                cell['source'] = new_source
                updated = True
                print("Fixed DeepJSCC cell.")
                break

    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook successfully updated with DeepJSCC fix.")
    else:
        print("Could not find the DeepJSCC cell.")
        
except Exception as e:
    print(f"Error modifying notebook: {e}")
