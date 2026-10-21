"""Build a self-contained free-Colab GPU compatibility notebook, not a training claim."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value


def main():
    source_manifest = next((ROOT / "Models/author_releases/SwinJSCC").glob("*.manifest.json"))
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    probe = (ROOT / "SharedCode/reproduction/runtime_probe_swin.py").read_text(encoding="utf-8")
    cells = [cell("markdown", "# SwinJSCC — free GPU compatibility pilot\n\n"
                  "This notebook uses the pinned, unchanged author base-model release and an existing C=96 checkpoint. "
                  "It checks strict checkpoint loading, native AWGN forward passes at training and full Kodak geometry, finite gradients, "
                  "optimizer-state loading and checkpoint round-trip. Inputs are synthetic; it does **not** claim new real training epochs or Kodak quality results.\n\n"
                  "Use **Runtime → Change runtime type → T4 GPU**, then connect. Free GPU availability is not guaranteed. "
                  "Do not select an upgrade or paid plan. The notebook does not purchase anything. "
                  "The author source is checked by commit and file hashes before and after execution.\n\n"
                  "Upload the existing AWGN `model.pth` only when the upload cell asks. Original local weights are preserved. "
                  "The optimizer probe writes a separately named synthetic checkpoint, which must never be counted as trained benchmark progress."),
             cell("code", "from pathlib import Path\nimport sys, subprocess, json, hashlib, io, zipfile\nimport requests\nimport torch\n"
                  "assert torch.cuda.is_available(), 'No free GPU connected. Select a free T4 runtime; do not buy a plan.'\n"
                  "ROOT = Path('/content/SemanticLosslessPilot')\nROOT.mkdir(parents=True, exist_ok=True)\n"
                  "hardware = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count()}\n"
                  "print(json.dumps(hardware, indent=2))\n"
                  "subprocess.run(['nvidia-smi'], check=True)\n"),
             cell("code", "deps = ROOT / 'SharedCode/reproduction/.cpu_dependencies'\n"
                  "subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(deps), '--no-deps', 'timm==0.6.13', 'pytorch-msssim==1.0.0'], check=True)\n"
                  "print('Pilot dependencies installed; existing PyTorch is not replaced.')\n"),
             cell("code", "manifest = " + repr(manifest) + "\n"
                  "source = ROOT / manifest['source_directory']\n"
                  "response = requests.get(manifest['archive_url'], timeout=(20,120))\nresponse.raise_for_status()\n"
                  "with zipfile.ZipFile(io.BytesIO(response.content)) as archive:\n"
                  "    for item in archive.infolist():\n"
                  "        if item.is_dir(): continue\n"
                  "        relative = Path(*Path(item.filename).parts[1:])\n"
                  "        assert '..' not in relative.parts and not relative.is_absolute()\n"
                  "        destination = source / relative\n"
                  "        destination.parent.mkdir(parents=True, exist_ok=True)\n"
                  "        destination.write_bytes(archive.read(item))\n"
                  "for name, digest in manifest['files_sha256'].items():\n"
                  "    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name\n"
                  "(source.parent / (manifest['commit'] + '.manifest.json')).write_text(json.dumps(manifest, indent=2))\n"
                  "print('Unchanged author source verified:', manifest['commit'])\n"),
             cell("code", "probe_source = " + repr(probe) + "\n"
                  "probe_path = ROOT / 'SharedCode/reproduction/runtime_probe_swin.py'\n"
                  "probe_path.parent.mkdir(parents=True, exist_ok=True)\nprobe_path.write_text(probe_source)\n"),
             cell("markdown", "## Existing checkpoint\n\nUpload the **AWGN C=96** checkpoint from the original SwinJSCC run. "
                  "This diagnostic requires strict loading; it will fail on the wrong architecture rather than skipping weights."),
             cell("code", "from google.colab import files\n"
                  "uploaded = files.upload()\n"
                  "assert len(uploaded) == 1, 'Upload exactly the AWGN model.pth checkpoint.'\n"
                  "CHECKPOINT = Path(next(iter(uploaded))).resolve()\n"
                  "assert CHECKPOINT.is_file()\n"
                  "del uploaded\n"
                  "print('Checkpoint file available:', CHECKPOINT.name, CHECKPOINT.stat().st_size, 'bytes')\n"),
             cell("code", "output = ROOT / 'Results/swin_gpu_runtime_probe'\n"
                  "result = subprocess.run([sys.executable, '-u', str(probe_path), '--root', str(ROOT), '--device', 'cuda', '--checkpoint', str(CHECKPOINT), '--output', str(output)], check=False)\n"
                  "report_path = output / 'report.json'\n"
                  "assert report_path.exists(), 'Probe did not produce a status report.'\n"
                  "report = json.loads(report_path.read_text())\nprint(json.dumps(report, indent=2))\n"
                  "assert result.returncode == 0 and report['status'] == 'PASS', 'GPU validation failed; inspect report before training.'\n"),
             cell("code", "# Retrieve actual diagnostic evidence even if later training is not started.\n"
                  "evidence = ROOT / 'swin_gpu_pilot_evidence.zip'\n"
                  "with zipfile.ZipFile(evidence, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                  "    archive.write(report_path, 'report.json')\n"
                  "    archive.writestr('hardware.json', json.dumps(hardware, indent=2))\n"
                  "    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)\n"
                  "    archive.writestr('environment.txt', freeze)\n"
                  "files.download(str(evidence))\n"
                  "print('Pilot completed. No DIV2K training epoch or Kodak metric was claimed.')\n")]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    output = ROOT / "Notebooks/Colab/SwinJSCC_Free_GPU_Preflight.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
