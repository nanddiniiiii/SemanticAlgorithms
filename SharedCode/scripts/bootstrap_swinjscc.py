"""Clone one audited SwinJSCC revision and apply minimal modern-runtime corrections.

The patch is intentionally narrow. It is not a fork and it is not applied to an
unknown upstream tree: the exact Git commit and source snippets are verified first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


UPSTREAM_URL = "https://github.com/semcomm/SwinJSCC.git"
UPSTREAM_COMMIT = "a6d0e6da53548976acbe9317839a077ef31f190f"
PATCH_VERSION = "stage1a-per-image-symbols-v2"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_git(arguments: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    if old not in content and new in content:
        return
    occurrences = content.count(old)
    if occurrences != 1:
        raise RuntimeError(
            f"Refusing to patch {path}: expected one audited source hunk, found {occurrences}."
        )
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def replace_expected(path: Path, old: str, new: str, expected_count: int) -> None:
    """Replace an audited repeated hunk, accepting a fully patched file on rerun."""

    content = path.read_text(encoding="utf-8")
    occurrences = content.count(old)
    if occurrences == 0 and content.count(new) >= expected_count:
        return
    if occurrences != expected_count:
        raise RuntimeError(
            f"Refusing to patch {path}: expected {expected_count} audited hunks, found {occurrences}."
        )
    path.write_text(content.replace(old, new), encoding="utf-8")


def ensure_upstream(root: Path) -> Path:
    upstream = root / "external" / "SwinJSCC"
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", UPSTREAM_URL, str(upstream)])
    actual_commit = run_git(["rev-parse", "HEAD"], cwd=upstream)
    if actual_commit != UPSTREAM_COMMIT:
        status = run_git(["status", "--porcelain"], cwd=upstream)
        if status:
            raise RuntimeError(
                "The existing upstream checkout is dirty and not at the required commit; refusing to overwrite it."
            )
        run_git(["fetch", "--tags", "origin"], cwd=upstream)
        run_git(["checkout", "--detach", UPSTREAM_COMMIT], cwd=upstream)
    return upstream


def apply_patch(upstream: Path) -> dict[str, str]:
    channel = upstream / "net" / "channel.py"
    encoder = upstream / "net" / "encoder.py"
    decoder = upstream / "net" / "decoder.py"
    marker = "# STAGE1A_MODERN_PATCH: per-image complex symbols and device-safe tensors"

    if marker not in channel.read_text(encoding="utf-8"):
        replace_once(
            channel,
            "        device = input_layer.get_device()\n",
            marker + "\n        device = input_layer.device\n",
        )
        replace_once(
            channel,
            "            noise = noise.to(input_layer.get_device())\n            h = h.to(input_layer.get_device())\n",
            "            noise = noise.to(input_layer.device)\n            h = h.to(input_layer.device)\n",
        )
        replace_once(
            channel,
            "        input_shape = channel_tx.shape\n        channel_in = channel_tx.reshape(-1)\n        L = channel_in.shape[0]\n        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j\n        channel_output = self.complex_forward(channel_in, chan_param)\n        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])\n        channel_output = channel_output.reshape(input_shape)\n",
            "        input_shape = channel_tx.shape\n        batch_size = channel_tx.shape[0]\n        channel_in = channel_tx.reshape(batch_size, -1)\n        L = channel_in.shape[1]\n        if L % 2 != 0:\n            raise ValueError('Each image must expose an even number of real channel values.')\n        # Keep real/imaginary pairing inside each image. Flattening the full batch\n        # mixes samples once DataParallel assigns more than one local image.\n        channel_in = channel_in[:, :L // 2] + channel_in[:, L // 2:] * 1j\n        channel_output = self.complex_forward(channel_in, chan_param)\n        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)], dim=1)\n        channel_output = channel_output.reshape(input_shape)\n",
        )

    mask_old_cuda = "            self.attn_mask = attn_mask.cuda()\n"
    mask_old_empty_module = "            self.attn_mask = attn_mask.to(next(self.parameters()).device)\n"
    mask_new = "            self.attn_mask = attn_mask.to(self.attn.relative_position_bias_table.device)\n"
    encoder_source = encoder.read_text(encoding="utf-8")
    if mask_new not in encoder_source:
        if mask_old_empty_module in encoder_source:
            encoder.write_text(encoder_source.replace(mask_old_empty_module, mask_new, 1), encoding="utf-8")
        else:
            replace_once(encoder, mask_old_cuda, mask_new)
    replace_once(encoder, "        device = x.get_device()\n", "        device = x.device\n")
    replace_expected(
        encoder,
        "            add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, rate)\n            c_indices = c_indices + add.int().cuda()\n            mask = torch.zeros(mask.size()).reshape(-1).cuda()\n",
        "            add = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, rate) * x.size()[2]\n            c_indices = c_indices + add.long()\n            mask = torch.zeros(mask.size(), device=x.device).reshape(-1)\n",
        expected_count=2,
    )
    replace_expected(
        decoder,
        "            device = x.get_device()\n",
        "            device = x.device\n",
        expected_count=2,
    )

    hashes = {}
    for path in (channel, encoder, decoder):
        hashes[str(path.relative_to(upstream))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (upstream / ".stage1a_patch.json").write_text(
        json.dumps(
            {"patch_version": PATCH_VERSION, "upstream_commit": UPSTREAM_COMMIT, "source_hashes": hashes},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    upstream = ensure_upstream(root)
    hashes = apply_patch(upstream)
    print(json.dumps({"commit": UPSTREAM_COMMIT, "patch_version": PATCH_VERSION, "hashes": hashes}, indent=2))


if __name__ == "__main__":
    main()
