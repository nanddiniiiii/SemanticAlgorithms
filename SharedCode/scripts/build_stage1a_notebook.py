"""Generate the one safe-to-rerun Kaggle Stage 1A notebook."""
from __future__ import annotations
import argparse
import uuid
from pathlib import Path

def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result

def build(project_ref: str) -> dict:
    md = lambda source: cell("markdown", source)
    py = lambda source: cell("code", source)
    cells = [
        md("# Stage 1A: corrected SwinJSCC, K=256 VQ, and AWGN evaluation\n\nOne Kaggle runner for the modular repository. It does not install or upgrade Kaggle's shared scientific stack.\n"),
        md("## 1. Clean a previous attempt\n"),
        py("""from pathlib import Path
import os
import shutil

os.chdir('/kaggle/working')
PROJECT_DIR = Path('/kaggle/working/SemanticSchedulerEnd2End')
STAGE1_OUTPUT = Path('/kaggle/working/stage1a_complete_output')
for target in (PROJECT_DIR, STAGE1_OUTPUT):
    if target.exists():
        print(f'Removing previous run: {target}')
        shutil.rmtree(target)
os.chdir('/kaggle/working')
print('Fresh working directory:', os.getcwd())
"""),
        md("## 2. Controls\n"),
        py(f"""from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_GIT_URL = 'https://github.com/saltypal/SemanticSchedulerEnd2End.git'
PROJECT_BRANCH = 'main'
# The notebook resolves this branch to an immutable commit immediately before cloning.
PROJECT_REF = None
UPSTREAM_SWINJSCC_COMMIT = 'a6d0e6da53548976acbe9317839a077ef31f190f'
BASE_EPOCHS = 200
SARA_EPOCHS = 300
BATCH_SIZE_PER_GPU = 8
EARLY_STOPPING_PATIENCE = 30
KMEANS_FIT_IMAGES = 512
VQ_EVALUATION_RATE = 96
JPEG_QUALITY_GRID = (20, 30, 40, 50, 60, 70, 80, 90)
TRAIN_ROOT = Path('/kaggle/input/notebooks/jagan028/div2k-dataset-generation-for-isr/Training/HR/hr_images')
VALIDATION_ROOT = Path('/kaggle/input/notebooks/jagan028/div2k-dataset-generation-for-isr/Validation/HR/hr_images')
KODAK_ROOT = Path('/kaggle/input/datasets/sherylmehta/kodak-dataset')

def run_command(*arguments: str, cwd: Path | None = None) -> None:
    print('+', ' '.join(arguments))
    subprocess.run(list(arguments), cwd=cwd, check=True)
"""),
        md("## 3. Clone source and audit the stock Kaggle runtime\n"),
        py("""import importlib
import torch

latest_ref_output = subprocess.check_output(
    ['git', 'ls-remote', PROJECT_GIT_URL, f'refs/heads/{PROJECT_BRANCH}'],
    text=True,
).strip()
if not latest_ref_output:
    raise RuntimeError(f'Could not resolve remote branch {PROJECT_BRANCH!r} from {PROJECT_GIT_URL}')
PROJECT_REF = latest_ref_output.split()[0]
print(f'Latest {PROJECT_BRANCH} commit: {PROJECT_REF}')

run_command('git', 'clone', '--no-checkout', PROJECT_GIT_URL, str(PROJECT_DIR), cwd=Path('/kaggle/working'))
run_command('git', 'checkout', '--detach', PROJECT_REF, cwd=PROJECT_DIR)
resolved_ref = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR, text=True).strip()
if resolved_ref != PROJECT_REF:
    raise RuntimeError(f'Project pin mismatch: expected {PROJECT_REF}, got {resolved_ref}')
run_command(sys.executable, 'scripts/bootstrap_swinjscc.py', cwd=PROJECT_DIR)
upstream_root = PROJECT_DIR / 'external' / 'SwinJSCC'
upstream_ref = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream_root, text=True).strip()
if upstream_ref != UPSTREAM_SWINJSCC_COMMIT:
    raise RuntimeError(f'Upstream pin mismatch: expected {UPSTREAM_SWINJSCC_COMMIT}, got {upstream_ref}')

src_path = str(PROJECT_DIR / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
importlib.invalidate_caches()
os.chdir(PROJECT_DIR)
STAGE1_OUTPUT.mkdir(parents=True, exist_ok=True)

# Do not run pip here. Kaggle provides the compiled scientific stack.
import numpy
import timm

runtime = {
    'python': sys.version, 'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(),
    'numpy': numpy.__version__, 'timm': timm.__version__,
}
print(json.dumps(runtime, indent=2))
if torch.cuda.device_count() < 2:
    raise RuntimeError('Enable Kaggle GPU x2 before running Stage 1A.')
print('Runtime audit passed; no global packages were changed.')
"""),
        md("## 4. Import project and validate inputs\n"),
        py("""import EncDecPipeline
from EncDecPipeline.Models.SwinJSCC.stage1_runner import (
    evaluate_jpeg_over_digital_awgn, evaluate_native_swinjscc,
    evaluate_vq_noiseless_ablation, evaluate_vq_over_digital_awgn,
    export_final_stage1_artifacts, fit_k256_codebook,
    resolve_required_datasets, run_base_then_sara_training,
    verify_kaggle_stage1a_runtime,
)
from EncDecPipeline.Models.SwinJSCC.swin_config import SwinJSCCConfig
from EncDecPipeline.Models.SwinJSCC.trainer import RealImageDataset, build_base_then_sara
from EncDecPipeline.Models.SwinJSCC.training_utils import mse_loss, verify_two_t4_data_parallel, wrap_data_parallel
from Evaluation.Reporting.plotting import plot_quality_vs_channel_uses
from Evaluation.Reporting.report_builder import write_run_report
from Evaluation.Reporting.result_table import write_result_table
from utils.seed import set_seed

set_seed(20260906)
verify_kaggle_stage1a_runtime()
data_provenance = resolve_required_datasets(TRAIN_ROOT, VALIDATION_ROOT, KODAK_ROOT)
config = SwinJSCCConfig()
(STAGE1_OUTPUT / 'startup.json').write_text(json.dumps({'runtime': runtime, 'data_provenance': data_provenance}, indent=2), encoding='utf-8')
print('Project import:', EncDecPipeline.__file__)
"""),
        md("## 5. Two-T4 preflight\n"),
        py("""plan = verify_two_t4_data_parallel()
preflight_adapter, _ = build_base_then_sara(config, 'external/SwinJSCC')
preflight_model = wrap_data_parallel(preflight_adapter.build_training_module(), plan)
preflight_optimizer = torch.optim.AdamW(preflight_model.parameters(), lr=1e-4)
preflight_dataset = RealImageDataset(TRAIN_ROOT, crop_size=256, training=True, seed=20260906)
real_images = torch.stack([preflight_dataset[0], preflight_dataset[1]]).cuda(non_blocking=True)
losses = []
for step in range(8):
    preflight_optimizer.zero_grad(set_to_none=True)
    reconstruction, _ = preflight_model(real_images, 10, config.base_fixed_c)
    loss = mse_loss(real_images, reconstruction)
    loss.backward()
    preflight_optimizer.step()
    loss_value = float(loss.detach().cpu())
    losses.append(loss_value)
    print(f'Preflight {step + 1}/8 | MSE={loss_value:.6f}')
if not all(torch.isfinite(torch.tensor(losses))):
    raise RuntimeError(f'Non-finite preflight loss: {losses}')
del preflight_model, preflight_adapter, preflight_optimizer
torch.cuda.empty_cache()
"""),
        md("## 6. Train and export\n"),
        py("""adapter, training_summary = run_base_then_sara_training(
    config=config, upstream_root='external/SwinJSCC',
    train_root=TRAIN_ROOT, validation_root=VALIDATION_ROOT,
    output_dir=STAGE1_OUTPUT, base_epochs=BASE_EPOCHS,
    sara_epochs=SARA_EPOCHS, batch_size_per_gpu=BATCH_SIZE_PER_GPU,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
)
print(json.dumps(training_summary, indent=2))
export_final_stage1_artifacts(adapter, STAGE1_OUTPUT / 'models' / 'SwinJSCC', data_provenance, training_summary)
"""),
        md("## 7. Fit K=256 latent VQ\n"),
        py("""vq = fit_k256_codebook(adapter, TRAIN_ROOT, rate=VQ_EVALUATION_RATE, max_images=KMEANS_FIT_IMAGES)
codebook_path = vq.save(STAGE1_OUTPUT / 'models' / 'Codebooks' / 'swinjscc_tx_k256.joblib')
print({'codebook': str(codebook_path), 'hash': vq.content_hash()})
"""),
        md("## 8. Evaluate\n"),
        py("""rows = []
rows.extend(evaluate_native_swinjscc(adapter, KODAK_ROOT, STAGE1_OUTPUT, config.snr_db_grid, config.rate_grid))
rows.extend(evaluate_vq_noiseless_ablation(adapter, vq, KODAK_ROOT, STAGE1_OUTPUT, rate=VQ_EVALUATION_RATE))
rows.extend(evaluate_vq_over_digital_awgn(adapter, vq, KODAK_ROOT, STAGE1_OUTPUT, config.snr_db_grid, rate=VQ_EVALUATION_RATE))
for quality in JPEG_QUALITY_GRID:
    rows.extend(evaluate_jpeg_over_digital_awgn(KODAK_ROOT, STAGE1_OUTPUT, config.snr_db_grid, quality))
metrics_path = write_result_table(rows, STAGE1_OUTPUT / 'metrics' / 'stage1a_metrics.csv')
successful_rows = [row for row in rows if row.get('psnr') is not None and row['method'] != 'vq_noiseless_ablation']
plot_quality_vs_channel_uses(successful_rows, STAGE1_OUTPUT / 'plots' / 'quality_vs_channel_uses.png')
write_run_report({'project_ref': PROJECT_REF, 'upstream_commit': UPSTREAM_SWINJSCC_COMMIT, 'metrics_csv': str(metrics_path), 'rows': len(rows), 'corrected_port': True}, STAGE1_OUTPUT / 'reports' / 'stage1a_report.md')
print('Evaluation complete:', metrics_path)
"""),
        md("## 9. Artifact audit\n"),
        py("""expected = [
    STAGE1_OUTPUT / 'models' / 'SwinJSCC' / 'stage1_swinjscc_full.pt',
    STAGE1_OUTPUT / 'models' / 'SwinJSCC' / 'stage1_swinjscc_encoder.pt',
    STAGE1_OUTPUT / 'models' / 'SwinJSCC' / 'stage1_swinjscc_decoder.pt',
    STAGE1_OUTPUT / 'models' / 'SwinJSCC' / 'stage1_manifest.json',
    STAGE1_OUTPUT / 'models' / 'Codebooks' / 'swinjscc_tx_k256.joblib',
    STAGE1_OUTPUT / 'metrics' / 'stage1a_metrics.csv',
]
missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise RuntimeError(f'Stage 1A output audit failed: {missing}')
print('Stage 1A complete:', STAGE1_OUTPUT)
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-ref", default="SET_TO_COMMITTED_SOURCE_REVISION")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks" / "Stage1A" / "01_stage1a_complete_kaggle.ipynb")
    arguments = parser.parse_args()
    import nbformat
    notebook = nbformat.from_dict(build(arguments.project_ref))
    for notebook_cell in notebook.cells:
        notebook_cell["id"] = uuid.uuid4().hex[:12]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, arguments.output)
    print(arguments.output)

if __name__ == "__main__":
    main()
