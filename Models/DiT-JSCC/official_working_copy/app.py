"""Polished Gradio demo for DiT-JSCC."""

from __future__ import annotations

import os
import threading
from html import escape
from pathlib import Path

import gradio as gr

from dit_jscc import DiTJSCCPipeline, check_environment
from dit_jscc.pipeline import DEFAULT_AUTOENCODER, DEFAULT_CHECKPOINTS, SUPPORTED_RATES


ROOT = Path(__file__).resolve().parent
PIPELINE_LOCK = threading.Lock()
ACTIVE_PIPELINE: DiTJSCCPipeline | None = None
ACTIVE_RESOLUTION: int | None = None

CSS = """
:root {
  --ink: #111827;
  --muted: #667085;
  --line: rgba(15, 23, 42, .09);
  --indigo: #4f46e5;
  --violet: #7c3aed;
}
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(circle at 10% 0%, rgba(99,102,241,.12), transparent 28rem),
    radial-gradient(circle at 90% 8%, rgba(14,165,233,.10), transparent 24rem),
    #f8fafc !important;
  color: var(--ink);
}
.main-wrap { padding: 18px 8px 44px; }
.hero {
  position: relative;
  overflow: hidden;
  padding: 46px 48px;
  border: 1px solid rgba(255,255,255,.5);
  border-radius: 28px;
  background: linear-gradient(135deg, #111827 0%, #312e81 52%, #4338ca 100%);
  box-shadow: 0 24px 70px rgba(49,46,129,.22);
  color: white;
}
.hero:after {
  content: "";
  position: absolute;
  width: 360px; height: 360px; right: -90px; top: -170px;
  border-radius: 999px;
  border: 72px solid rgba(255,255,255,.06);
}
.eyebrow { font: 700 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .15em; text-transform: uppercase; color: #c7d2fe; }
.hero h1 { margin: 12px 0 10px; font-size: clamp(38px, 5.5vw, 72px); line-height: .98; letter-spacing: -.055em; font-weight: 800; }
.hero h1 span { background: linear-gradient(90deg,#a5b4fc,#7dd3fc); -webkit-background-clip: text; color: transparent; }
.hero p { max-width: 760px; margin: 0; color: #e0e7ff; font-size: 17px; line-height: 1.65; }
.badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }
.badge { padding:7px 11px; border-radius:999px; border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.09); font-size:12px; font-weight:650; color:white; }
.section-title { margin: 34px 0 12px; }
.section-title h2 { margin:0; font-size:26px; letter-spacing:-.035em; }
.section-title p { margin:6px 0 0; color:var(--muted); }
.panel, .gr-panel {
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  background: rgba(255,255,255,.86) !important;
  box-shadow: 0 12px 30px rgba(15,23,42,.055) !important;
}
.image-shell { min-height: 360px; }
.run-button {
  min-height: 48px !important;
  border: 0 !important;
  border-radius: 14px !important;
  background: linear-gradient(90deg,var(--indigo),var(--violet)) !important;
  box-shadow: 0 12px 24px rgba(79,70,229,.2) !important;
  font-weight: 750 !important;
}
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:16px; }
.feature-card { padding:18px; border-radius:18px; border:1px solid var(--line); background:rgba(255,255,255,.72); }
.feature-card strong { display:block; margin-bottom:6px; font-size:15px; }
.feature-card span { color:var(--muted); font-size:13px; line-height:1.55; }
.status-ok { padding:12px 14px; border-radius:12px; background:#ecfdf3; color:#027a48; border:1px solid #abefc6; }
.status-warn { padding:12px 14px; border-radius:12px; background:#fffaeb; color:#b54708; border:1px solid #fedf89; }
.status-error { padding:12px 14px; border-radius:12px; background:#fef3f2; color:#b42318; border:1px solid #fecdca; }
.footer-note { text-align:center; color:var(--muted); font-size:12px; margin-top:26px; }
@media (max-width: 760px) {
  .hero { padding:32px 24px; border-radius:22px; }
  .feature-grid { grid-template-columns:1fr; }
}
"""


def _setup_status(resolution: int) -> str:
    report = check_environment(int(resolution))
    if report["ready"]:
        device = report.get("cuda_device") or "CPU"
        return f'<div class="status-ok">Ready · {device}</div>'
    missing = []
    if not report["checkpoint_found"]:
        missing.append(DEFAULT_CHECKPOINTS[int(resolution)].name)
    if not report["autoencoder_found"]:
        missing.append(DEFAULT_AUTOENCODER.name)
    details = ", ".join(missing) if missing else "Python dependencies"
    return (
        '<div class="status-warn"><b>Setup required</b> · Missing '
        f'{details}. See <code>checkpoints/README.md</code>.</div>'
    )


def _rate_controls(resolution: int):
    resolution = int(resolution)
    choices = SUPPORTED_RATES[resolution]
    defaults = (8, 8) if resolution == 256 else (32, 32)
    return (
        gr.Dropdown(choices=list(choices["semantic"]), value=defaults[0]),
        gr.Dropdown(choices=list(choices["detail"]), value=defaults[1]),
        _setup_status(resolution),
    )


def _get_pipeline(resolution: int) -> DiTJSCCPipeline:
    global ACTIVE_PIPELINE, ACTIVE_RESOLUTION
    resolution = int(resolution)
    if ACTIVE_PIPELINE is None or ACTIVE_RESOLUTION != resolution:
        if ACTIVE_PIPELINE is not None:
            ACTIVE_PIPELINE.close()
        ACTIVE_PIPELINE = DiTJSCCPipeline(resolution=resolution)
        ACTIVE_RESOLUTION = resolution
    return ACTIVE_PIPELINE


def reconstruct(
    image,
    resolution,
    channel_type,
    snr,
    semantic_rate,
    detail_rate,
    sampler,
    steps,
    guidance,
    seed,
    progress=gr.Progress(),
):
    if image is None:
        return None, {}, '<div class="status-error">Upload an image first.</div>'
    sampler_key = {
        "DPM-Solver · fast": "dpm_solver",
        "Euler ODE · deterministic": "euler_ode",
        "Euler SDE · paper setting": "euler_sde",
    }[sampler]
    try:
        progress(0.05, desc="Loading DiT-JSCC")
        with PIPELINE_LOCK:
            pipeline = _get_pipeline(int(resolution))
            progress(0.18, desc="Encoding semantics and channel features")
            result = pipeline(
                image,
                channel_type=str(channel_type).lower(),
                snr_db=float(snr),
                semantic_rate=int(semantic_rate),
                detail_rate=int(detail_rate),
                sampler=sampler_key,
                steps=int(steps),
                guidance_scale=float(guidance),
                seed=int(seed),
            )
        progress(1.0, desc="Reconstruction complete")
        meta = result.metadata()
        cbr_inv = round(1 / meta["cbr"], 2) if meta["cbr"] else float("inf")
        status = (
            '<div class="status-ok"><b>Reconstruction complete</b> · '
            f'{meta["elapsed_seconds"]:.2f}s · CBR ≈ 1/{cbr_inv:g}</div>'
        )
        return result.image, meta, status
    except Exception as exc:
        message = escape(str(exc)).replace("\n", "<br>")
        return None, {}, f'<div class="status-error"><b>Could not run inference.</b><br>{message}</div>'


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    )
    with gr.Blocks(theme=theme, css=CSS, title="DiT-JSCC Studio") as demo:
        gr.HTML(
            """
            <div class="main-wrap">
              <section class="hero">
                <div class="eyebrow">Generative semantic communication</div>
                <h1>DiT-<span>JSCC</span></h1>
                <p>Rethinking Deep JSCC with diffusion transformers and semantic
                representations. Explore robust, semantics-prioritized image transmission
                under extreme bandwidth and wireless channel conditions.</p>
                <div class="badges">
                  <span class="badge">Diffusion Transformer</span>
                  <span class="badge">Dual-Branch JSCC</span>
                  <span class="badge">DINOv2 Semantics</span>
                  <span class="badge">AWGN Channel</span>
                </div>
              </section>
              <div class="feature-grid">
                <div class="feature-card"><strong>Semantic first</strong><span>High-level VFM representations preserve objects and global structure when channel resources are scarce.</span></div>
                <div class="feature-card"><strong>Coarse to fine</strong><span>Semantic and detail cues enter different DiT depths to progressively restore structure, texture, and edges.</span></div>
                <div class="feature-card"><strong>Channel aware</strong><span>Interactively study SNR, semantic bandwidth, detail bandwidth, sampler, and random-seed effects.</span></div>
              </div>
            </div>
            """
        )

        gr.HTML(
            '<div class="section-title"><h2>Interactive transmission lab</h2>'
            '<p>Upload an image, configure the simulated link, and reconstruct it at the receiver.</p></div>'
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes="panel"):
                input_image = gr.Image(
                    label="Source image",
                    type="pil",
                    height=420,
                    elem_classes="image-shell",
                )
                example_path = ROOT / "examples" / "example_input.png"
                if example_path.is_file():
                    gr.Examples(
                        examples=[[str(example_path)]],
                        inputs=[input_image],
                        label="Try the paper example",
                    )
            with gr.Column(scale=6, elem_classes="panel"):
                output_image = gr.Image(
                    label="Receiver reconstruction",
                    type="pil",
                    height=420,
                    interactive=False,
                    elem_classes="image-shell",
                )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="panel"):
                gr.Markdown("#### Link configuration")
                resolution = gr.Radio(
                    [256, 512], value=256, label="Resolution", info="Model checkpoint"
                )
                channel_type = gr.Radio(
                    ["AWGN", "Rayleigh"], value="AWGN", label="Channel"
                )
                snr = gr.Slider(-10, 20, value=0, step=1, label="SNR (dB)")
                with gr.Row():
                    semantic_rate = gr.Dropdown(
                        list(SUPPORTED_RATES[256]["semantic"]),
                        value=8,
                        label="Semantic channels",
                    )
                    detail_rate = gr.Dropdown(
                        list(SUPPORTED_RATES[256]["detail"]),
                        value=8,
                        label="Detail channels",
                    )
            with gr.Column(scale=4, elem_classes="panel"):
                gr.Markdown("#### Generative decoder")
                sampler = gr.Dropdown(
                    [
                        "DPM-Solver · fast",
                        "Euler ODE · deterministic",
                        "Euler SDE · paper setting",
                    ],
                    value="DPM-Solver · fast",
                    label="Sampler",
                )
                steps = gr.Slider(5, 200, value=20, step=5, label="Sampling steps")
                guidance = gr.Slider(0, 5, value=2, step=0.1, label="Guidance scale")
                seed = gr.Number(value=1000, precision=0, label="Seed")
            with gr.Column(scale=4, elem_classes="panel"):
                gr.Markdown("#### Run & inspect")
                setup_status = gr.HTML(_setup_status(256))
                run = gr.Button("Transmit & reconstruct", variant="primary", elem_classes="run-button")
                status = gr.HTML('<div class="status-warn">Waiting for an input image.</div>')
                metadata = gr.JSON(label="Run metadata", open=False)

        resolution.change(
            _rate_controls,
            inputs=[resolution],
            outputs=[semantic_rate, detail_rate, setup_status],
        )
        run.click(
            reconstruct,
            inputs=[
                input_image,
                resolution,
                channel_type,
                snr,
                semantic_rate,
                detail_rate,
                sampler,
                steps,
                guidance,
                seed,
            ],
            outputs=[output_image, metadata, status],
        )

        gr.HTML(
            '<div class="section-title"><h2>How it works</h2>'
            '<p>A semantics-detail transmitter meets a coarse-to-fine conditional DiT receiver.</p></div>'
        )
        with gr.Row():
            architecture = ROOT / "docs" / "assets" / "architecture.png"
            gr.Image(
                value=str(architecture),
                label="DiT-JSCC system architecture",
                interactive=False,
                show_download_button=False,
            )
        with gr.Accordion("Research results from the paper", open=False):
            with gr.Row():
                gr.Image(
                    value=str(ROOT / "docs" / "assets" / "semantic-results.png"),
                    label="Semantic and perceptual consistency",
                    interactive=False,
                    show_download_button=False,
                )
                gr.Image(
                    value=str(ROOT / "docs" / "assets" / "visual-results-512.png"),
                    label="512 × 512 qualitative results",
                    interactive=False,
                    show_download_button=False,
                )
        gr.HTML(
            '<div class="footer-note">DiT-JSCC · Research preview · '
            '<a href="https://github.com/semcomm/DiTJSCC" target="_blank">GitHub</a></div>'
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        allowed_paths=[str(ROOT / "docs" / "assets"), str(ROOT / "examples")],
    )
