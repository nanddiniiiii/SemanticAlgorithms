<div align="center">

# DiT-JSCC

### 基于扩散 Transformer 与语义表征的新一代生成式联合信源信道编码

[English](README.md) · [推理指南](docs/INFERENCE.md)

</div>

![DiT-JSCC 系统架构](docs/assets/architecture.png)

## 项目简介

DiT-JSCC 面向极低带宽、低信噪比条件下的图像传输。与只追求像素重建或仅在接收端增加生成模型的方法不同，本项目联合设计发送端表征与生成式接收端：

- 使用 DINOv2 提取高层语义，通过语义分支优先传输物体、布局和全局结构；
- 使用 SwinJSCC 细节分支传输纹理、边缘等低层信息；
- 在 DiT 的不同深度注入两类条件，形成由粗到细的生成式解码；
- 支持语义/细节带宽独立调节，并在论文中提出受 Kolmogorov 复杂度启发的免训练自适应带宽分配方法。

## 快速开始

```bash
git clone https://github.com/semcomm/DiTJSCC.git
cd DiTJSCC
conda create -n ditjscc python=3.10 -y
conda activate ditjscc

# 请先安装与你的 CUDA 版本匹配的 PyTorch
pip install -e .
```

仓库当前不包含模型参数。后续发布权重时，按以下预留结构放入：

```text
checkpoints/
├── autoencoder_kl.pth
├── dit_jscc_256.pth
└── dit_jscc_512.pth
```

检查环境：

```bash
python infer.py --check --resolution 256
```

单图推理：

```bash
python infer.py \
  --input examples/example_input.png \
  --output outputs/demo \
  --resolution 256 \
  --snr 0 \
  --semantic-rate 8 \
  --detail-rate 8 \
  --steps 20
```

启动可视化界面：

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。界面可控制分辨率、SNR、两路带宽、采样器、采样步数、CFG scale 与随机种子，并输出实际 CBR 和运行时间。

## 结果概览

![512 分辨率定性结果](docs/assets/visual-results-512.png)

论文在 ImageNet 上从感知一致性（LPIPS、DISTS）、语义一致性（CLIP、DreamSim、DINOv2）和真实性（FID）三个维度进行评估。在 256 × 256、CBR=1/48、0 dB AWGN 设置下，DiT-JSCC 报告了 0.166 LPIPS、0.963 CLIP、0.958 DINOv2 与 60 FID。完整实验设置、基线和 Rayleigh 信道结果请参阅论文。

## 目录说明

- `dit_jscc/`：面向公开使用的稳定推理 API；
- `app.py`：Gradio Web Studio；
- `infer.py`：单图/文件夹批量推理；
- `configs/`：去除本机路径后的 256/512 推理配置；
- `inference_core/`：与模型参数兼容的最小推理核心；
- `checkpoints/`：预留的模型参数目录；
- `docs/`：推理文档和项目展示图片；
- `tests/`：轻量公开接口测试。

本目录是独立、可直接上传的推理项目。完整训练/评估代码、论文
LaTeX、论文 PDF 和本地实验文件均保留在外层研究目录，不会被加入
此 Git 仓库。

更完整的参数、预处理、复现和排错说明见 [推理指南](docs/INFERENCE.md)。引用信息与许可证请参阅英文首页。
