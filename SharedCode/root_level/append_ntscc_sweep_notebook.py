"""Append executed, result-loading cells without changing existing notebook cells."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "ImageFeatures_DeepJSCC_NTSCC_FINAL_REPAIRED.ipynb"
MARKER = "## 8. Rate-conditioned NTSCC PQ/RVQ sweep"

notebook = nbformat.read(PATH, as_version=4)
if any(MARKER in "".join(cell.get("source", "")) for cell in notebook.cells):
    raise RuntimeError("Sweep section already exists; refusing to duplicate it")

markdown = nbformat.v4.new_markdown_cell(
    MARKER
    + "\n\nThe original/frozen NTSCC files and the existing benchmark cells above are unchanged. "
      "This section loads the separately measured rate-conditioned wrapper sweep. Calibration uses "
      "500 CIFAR-10 training images plus all 23 non-test Kodak images; `kodim08` is excluded."
)
code = nbformat.v4.new_code_cell(
    "from pathlib import Path\n"
    "import json\n"
    "import pandas as pd\n"
    "from IPython.display import display, Image as DisplayImage\n"
    "sweep_dir = Path.cwd() / 'benchmark_outputs' / 'ntscc_rate_conditioned_sweep'\n"
    "sweep_results = pd.read_csv(sweep_dir / 'sweep_results.csv')\n"
    "identity_check = json.loads((sweep_dir / 'identity_verification.json').read_text(encoding='utf-8'))\n"
    "summary = json.loads((sweep_dir / 'summary.json').read_text(encoding='utf-8'))\n"
    "display(sweep_results[[\n"
    "    'configuration', 'block_size', 'K', 'normalized', 'rvq_stages',\n"
    "    'psnr_db', 'ssim', 'latent_mse_active', 'total_transmitted_bits',\n"
    "    'bpp', 'fit_latency_s', 'evaluation_latency_s', 'status'\n"
    "]].sort_values(['bpp', 'psnr_db'], ascending=[True, False]))\n"
    "display(pd.DataFrame([identity_check]))\n"
    "print('Selected Pareto-knee configuration:', summary['best_configuration']['configuration'])\n"
    "display(DisplayImage(filename=str(sweep_dir / 'rate_distortion_sweep.png')))\n"
    "display(DisplayImage(filename=str(sweep_dir / 'best_reconstruction.png')))\n"
)

mini = nbformat.v4.new_notebook(cells=[code], metadata=notebook.metadata)
executed = NotebookClient(mini, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
notebook.cells.extend([markdown, executed.cells[0]])
nbformat.write(notebook, PATH)
