import json
from pathlib import Path


path = Path("ImageFeatures_DeepJSCC_NTSCC_FINAL_REPAIRED.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))

notebook["cells"][21]["source"] = [
    "## 7. Consolidated measured comparison and requested figures\n",
    "\n",
    "All values below come from this execution. `N/A` is retained where a bit count or latent representation is not mathematically defined.\n",
]
notebook["cells"][22]["source"] = [
    "consolidated = session.consolidated_table()\n",
    "display(consolidated.fillna('N/A'))\n",
    "display(DisplayImage(filename=str(session.output_dir / 'presentation_reconstruction_quality.png')))\n",
    "display(DisplayImage(filename=str(session.output_dir / 'presentation_channel_usage.png')))\n",
    "display(DisplayImage(filename=str(session.output_dir / 'presentation_channel_resource_saving.png')))\n",
    "display(DisplayImage(filename=str(session.output_dir / 'native_reconstructions.png')))\n",
    "display(DisplayImage(filename=str(session.output_dir / 'codebook_reconstructions.png')))\n",
]

for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
