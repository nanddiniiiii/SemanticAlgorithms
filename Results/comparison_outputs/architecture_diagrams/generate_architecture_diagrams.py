"""Generate architecture diagrams for the validated DeepJSCC/NTSCC paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent

COLORS = {
    "io": "#edf2f7",
    "learned": "#dbeafe",
    "analog": "#dcfce7",
    "digital": "#f3e8ff",
    "side": "#ffedd5",
    "packet": "#fef3c7",
}


def box(ax, x, y, w, h, text, kind="learned", fs=9, lw=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=COLORS[kind], edgecolor="#334155", linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return (x, y, w, h)


def arrow(ax, start, end, color="#334155", dashed=False, lw=1.6, rad=0.0):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def link(ax, left, right, color="#334155", dashed=False, rad=0.0):
    arrow(
        ax,
        (left[0] + left[2], left[1] + left[3] / 2),
        (right[0], right[1] + right[3] / 2),
        color=color, dashed=dashed, rad=rad,
    )


def setup(ax, title, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.905, subtitle, ha="center", va="top", fontsize=9.5, color="#475569")


def legend(ax, y=0.03, fs=8):
    items = [
        ("learned", "learned transform"),
        ("analog", "continuous-symbol channel"),
        ("digital", "codebook/digital PHY"),
        ("side", "side information/calibration"),
    ]
    x = 0.19
    for kind, label in items:
        ax.add_patch(FancyBboxPatch((x, y), 0.022, 0.026, boxstyle="round,pad=0.002", facecolor=COLORS[kind], edgecolor="#64748b"))
        ax.text(x + 0.028, y + 0.013, label, va="center", fontsize=fs)
        x += 0.19


def deepjscc_native(ax):
    setup(
        ax,
        "DeepJSCC — native continuous-symbol architecture",
        "Validated notebook path: 128×128 tile inference, real latent paired into complex channel uses",
    )
    y, h, w = 0.50, 0.20, 0.085
    xs = [0.015, 0.123, 0.231, 0.339, 0.447, 0.555, 0.663, 0.771, 0.879]
    texts = [
        ("RGB image\n[0,1]", "io"),
        ("Reflect-pad\n+ 128×128 tiles", "io"),
        ("CNN encoder\n5 conv + PReLU", "learned"),
        ("Continuous\nreal latent\nC=19/tile", "learned"),
        ("I/Q pairing\n+ power norm", "analog"),
        ("Complex AWGN\nEs/N0", "analog"),
        ("I/Q unpair\nreceived latent", "analog"),
        ("CNN decoder\ntranspose conv", "learned"),
        ("Stitch + crop\nreconstruction", "io"),
    ]
    nodes = [box(ax, x, y, w, h, t, k, fs=8.3) for x, (t, k) in zip(xs, texts)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        link(ax, a, b)
    ax.text(
        0.5, 0.35,
        "Over-air resource: complex channel uses / CBR   •   No payload bits or entropy-coded file",
        ha="center", fontsize=10, color="#166534",
    )
    ax.text(
        0.5, 0.27,
        "Notebook measurement: 191,748 complex uses, CBR 0.16255 at 10 dB (checkpoint trained at 200 dB)",
        ha="center", fontsize=9, color="#475569",
    )
    legend(ax)


def deepjscc_codebook(ax):
    setup(
        ax,
        "DeepJSCC — K=256 codebook + digital PHY",
        "Encoder/decoder frozen; spatial latent tokens become 8-bit codebook indices",
    )
    y, h, w = 0.53, 0.20, 0.095
    xs = [0.012, 0.125, 0.238, 0.351, 0.464, 0.577, 0.690, 0.803, 0.904]
    texts = [
        ("RGB +\n128×128 tiling", "io"),
        ("Frozen CNN\nencoder", "learned"),
        ("Tokenize latent\n[B,N,C]", "learned"),
        ("Nearest centroid\nK=256\n8 bits/token", "digital"),
        ("Packet\nindices + shape\nmetadata + CRC", "packet"),
        ("5G LDPC\n+ QPSK", "digital"),
        ("AWGN + demod\nLDPC + CRC", "digital"),
        ("Shared K=256\ndequantize\n+ reshape", "digital"),
        ("Frozen decoder\n+ stitch/crop", "learned"),
    ]
    widths = [w] * 8 + [0.084]
    nodes = [box(ax, x, y, ww, h, t, k, fs=7.9) for x, ww, (t, k) in zip(xs, widths, texts)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        link(ax, a, b)
    calibration = box(ax, 0.20, 0.20, 0.22, 0.105, "CIFAR-10 training-split latents\nfit separate MiniBatchKMeans K=256", "side", fs=8.5)
    arrow(ax, (calibration[0] + calibration[2], calibration[1] + calibration[3] / 2), (nodes[3][0] + nodes[3][2] / 2, nodes[3][1]), color="#c2410c", dashed=True, rad=-0.15)
    arrow(ax, (calibration[0] + calibration[2], calibration[1] + calibration[3] / 2), (nodes[7][0] + nodes[7][2] / 2, nodes[7][1]), color="#c2410c", dashed=True, rad=-0.28)
    ax.text(0.51, 0.37, "Same versioned codebook is shared by transmitter and receiver; storage is not charged per image", ha="center", fontsize=9, color="#9a3412")
    ax.text(0.51, 0.10, "Measured packet: 161,632 bits → 323,584 coded bits → 161,792 complex uses (CBR 0.13715)", ha="center", fontsize=9, color="#475569")
    legend(ax, y=0.025)


def ntscc_native(ax):
    setup(
        ax,
        "NTSCC — native content-adaptive continuous-symbol architecture",
        "Exact adapter path: nonlinear transform + entropy probabilities drive per-token variable-rate JSCC",
    )
    y, h, w = 0.52, 0.20, 0.092
    xs = [0.012, 0.122, 0.232, 0.342, 0.452, 0.562, 0.672, 0.782, 0.892]
    texts = [
        ("RGB image\n[0,1]", "io"),
        ("Nonlinear\nanalysis g_a", "learned"),
        ("Hyperprior +\nentropy model\nP(y|z)", "side"),
        ("Rate-adaptive\nJSCC encoder f_e", "learned"),
        ("Masked continuous\nlatent s + mask\n+ rate_index", "learned"),
        ("Select active real\nvalues + I/Q", "analog"),
        ("Complex AWGN\ncontent-dependent uses", "analog"),
        ("Scatter by mask\nJSCC decoder f_d", "learned"),
        ("Synthesis g_s\nreconstruction", "learned"),
    ]
    nodes = [box(ax, x, y, w, h, t, k, fs=7.8) for x, (t, k) in zip(xs, texts)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        link(ax, a, b)
    # Direct rate-index side path used by the validated native benchmark.
    arrow(
        ax,
        (nodes[4][0] + nodes[4][2] / 2, nodes[4][1]),
        (nodes[7][0] + nodes[7][2] / 2, nodes[7][1]),
        color="#c2410c", dashed=True, rad=0.18,
    )
    ax.text(
        0.63, 0.31,
        "rate_index supplied directly to decoder in native benchmark\n(4 bits/token are reported but not transmitted)",
        ha="center", fontsize=8.5, color="#9a3412",
    )
    ax.text(0.5, 0.19, "Entropy bpp is diagnostic—not radio bits. Native resource is active complex channel uses/CBR.", ha="center", fontsize=9, color="#166534")
    ax.text(0.5, 0.11, "Measured: 36,126 complex uses, CBR 0.03062 at 10 dB", ha="center", fontsize=9, color="#475569")
    legend(ax, y=0.025)


def ntscc_codebook(ax):
    setup(
        ax,
        "NTSCC — active-prefix PQ16 K=256 codebook + digital PHY",
        "Improved measured variant: quantize only the officially active prefix of each spatial token",
    )
    y, h, w = 0.54, 0.20, 0.092
    xs = [0.010, 0.120, 0.230, 0.340, 0.450, 0.560, 0.670, 0.780, 0.890]
    texts = [
        ("RGB image", "io"),
        ("g_a + hyperprior\nentropy probabilities", "side"),
        ("Rate-adaptive f_e\ns_masked + rate_index", "learned"),
        ("Active prefix only\nsplit into 16-D blocks", "learned"),
        ("Nearest centroid\nPQ16 K=256\n8 bits/block", "digital"),
        ("Packet: indices +\n4-bit rate map +\nmetadata + CRC", "packet"),
        ("5G LDPC/QPSK\nAWGN + CRC", "digital"),
        ("Shared K=256\ndequantize blocks\nrestore mask/rates", "digital"),
        ("f_d → g_s\nreconstruction", "learned"),
    ]
    nodes = [box(ax, x, y, w, h, t, k, fs=7.65) for x, (t, k) in zip(xs, texts)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        link(ax, a, b)
    calibration = box(ax, 0.20, 0.20, 0.25, 0.105, "CIFAR-10 train images 0–63\nactive 16-D blocks → MiniBatchKMeans K=256", "side", fs=8.2)
    arrow(ax, (calibration[0] + calibration[2], calibration[1] + calibration[3] / 2), (nodes[4][0] + nodes[4][2] / 2, nodes[4][1]), color="#c2410c", dashed=True, rad=-0.12)
    arrow(ax, (calibration[0] + calibration[2], calibration[1] + calibration[3] / 2), (nodes[7][0] + nodes[7][2] / 2, nodes[7][1]), color="#c2410c", dashed=True, rad=-0.27)
    ax.text(0.69, 0.35, "Unlike native NTSCC, the rate map is inside the transmitted packet", ha="center", fontsize=8.7, color="#9a3412")
    ax.text(0.51, 0.11, "Measured packet: 42,440 bits → 86,016 coded bits → 43,008 complex uses (CBR 0.03646)", ha="center", fontsize=9, color="#475569")
    ax.text(0.77, 0.21, "Generic baseline alternative:\none K=256 centroid for each\nzero-padded 256-D token", ha="center", fontsize=8, color="#6b21a8", bbox={"facecolor": "#faf5ff", "edgecolor": "#c4b5fd", "boxstyle": "round,pad=0.4"})
    legend(ax, y=0.025)


DRAWERS = [
    ("01_deepjscc_native", deepjscc_native),
    ("02_deepjscc_with_codebook", deepjscc_codebook),
    ("03_ntscc_native", ntscc_native),
    ("04_ntscc_with_codebook", ntscc_codebook),
]


def save_single(name, drawer):
    fig, ax = plt.subplots(figsize=(16, 6.2))
    drawer(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_overview():
    fig, axes = plt.subplots(2, 2, figsize=(22, 12.5))
    for ax, (_, drawer) in zip(axes.flat, DRAWERS):
        drawer(ax)
    fig.suptitle("DeepJSCC and NTSCC architectures — native vs K=256 codebook", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=1.0)
    fig.savefig(OUT / "00_all_architectures_overview.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "00_all_architectures_overview.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, drawer in DRAWERS:
        save_single(name, drawer)
    save_overview()
    print(f"Wrote architecture diagrams to {OUT}")


if __name__ == "__main__":
    main()
