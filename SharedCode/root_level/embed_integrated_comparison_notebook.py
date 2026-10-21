"""Embed the wrapper and executed integrated comparison into the existing notebook."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "ImageFeatures_DeepJSCC_NTSCC_FINAL_REPAIRED.ipynb"
MARKER = "## 9. Integrated rate-conditioned NTSCC codebook comparison"

notebook = nbformat.read(NOTEBOOK, as_version=4)
if any(MARKER in str(cell.get("source", "")) for cell in notebook.cells):
    raise RuntimeError("Integrated comparison section is already embedded")

wrapper_source = (ROOT / "semcom_benchmark" / "ntscc_rate_conditioned_vq.py").read_text(encoding="utf-8")
runner_source = (ROOT / "run_integrated_ntscc_comparison.py").read_text(encoding="utf-8")
runner_source = runner_source.replace(
    "from semcom_benchmark.ntscc_rate_conditioned_vq import RateConditionedConfig, RateConditionedPQ\n",
    "",
)
runner_source = runner_source.replace(
    'ROOT = Path(__file__).resolve().parent',
    'ROOT = Path.cwd().resolve()',
)

markdown = nbformat.v4.new_markdown_cell(
    MARKER
    + "\n\nThe wrapper implementation and its integrated PHY evaluation are included below. "
      "The original NTSCC implementation remains frozen. The selected configuration is the measured "
      "Pareto-knee rate-conditioned PQ4 K=256 wrapper. All other rows are unchanged measured baselines "
      "from this notebook's earlier execution."
)
wrapper_cell = nbformat.v4.new_code_cell(wrapper_source)
runner_cell = nbformat.v4.new_code_cell(runner_source)
display_cell = nbformat.v4.new_code_cell(
    "from IPython.display import display, Image as DisplayImage\n"
    "integrated_dir = Path.cwd() / 'benchmark_outputs' / 'integrated_rate_conditioned_comparison'\n"
    "integrated_kpis = pd.read_csv(integrated_dir / 'all_methods_kpi_table.csv', keep_default_na=False)\n"
    "display(integrated_kpis)\n"
    "display(DisplayImage(filename=str(integrated_dir / 'all_methods_quality_ssim_lpips.png')))\n"
    "display(DisplayImage(filename=str(integrated_dir / 'all_methods_channel_usage.png')))\n"
    "display(DisplayImage(filename=str(integrated_dir / 'all_learned_reconstructions.png')))\n"
)

mini = nbformat.v4.new_notebook(
    cells=[wrapper_cell, runner_cell, display_cell],
    metadata=notebook.metadata,
)
executed = NotebookClient(
    mini,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
notebook.cells.extend([markdown, *executed.cells])
nbformat.write(notebook, NOTEBOOK)
