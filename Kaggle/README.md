# Kaggle accounts

Account A is `satyapaladugu` using the existing Windows Kaggle CLI authentication.
Account B is `satyapaladuguplus` using the WSL CLI installed at
`/home/Satya/.venvs/semantic-kaggle/bin/kaggle`. No credentials are stored here.

Account B legacy configuration directory:
`/home/Satya/.config/semantic-lossless/kaggle/account_b` (mode 0700).
The installed Kaggle 2.2.4 OAuth implementation stores its refresh credentials at
WSL `~/.kaggle/credentials.json`; `KAGGLE_CONFIG_DIR` alone does not relocate that
OAuth file. Windows A and WSL B therefore use separate credential stores. Do not
sign account A into the WSL default OAuth store. The launcher checks the expected
username before performing any command.

```text
python scripts/kaggle_account.py A kernels list --mine
python scripts/kaggle_account.py B kernels list --mine
```

To authenticate B locally:

```bash
KAGGLE_CONFIG_DIR="$HOME/.config/semantic-lossless/kaggle/account_b" \
  "$HOME/.venvs/semantic-kaggle/bin/kaggle" auth login --no-launch-browser
```

Use the browser authorization URL and enter its code directly into that terminal.
Never put tokens or verification codes into notebooks, source control, or chat.
Authentication is complete only after an authenticated API call confirms B.

The two `kernel-metadata.json` files request `NvidiaTeslaT4`, documented as T4 x2
by the [official Kaggle metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
The runtime still checks actual device availability before dual-GPU training.
Metadata alone does not prove quota, allocation, dataset access, or successful jobs.
Native dataset and asset source IDs must be populated before uploads. No long jobs
are launched by generating these files.

For uploads, pass `--accelerator NvidiaTeslaT4` explicitly. The WSL CLI accepted
GPU metadata but launched SQ-GAN versions 1-3 on CPU; version 4 explicitly requests
the accelerator. Every executable evaluation now checks CUDA before model load.

```text
python scripts/kaggle_account.py B kernels push -p /mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticLossless/Semantic_Lossless/Kaggle/sqgan_evaluation --accelerator NvidiaTeslaT4 --timeout 3600
```

WSL Firefox login: use `BROWSER='/home/Satya/.local/opt/firefox/firefox %s &'`
with `kaggle auth login --force`. The trailing background marker lets the OAuth
callback listener start while Firefox remains open. An ordinary foreground
browser command can block listener startup and cause localhost connection failure.
