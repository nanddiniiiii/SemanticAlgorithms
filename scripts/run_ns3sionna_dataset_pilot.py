"""Run a small ns-3/Sionna-RT bridge smoke and sample-load each planned image source.

This is a feasibility check only. It downloads one permitted sample file per
source through KaggleHub, decodes it, and records metadata/hashes without
copying a full dataset into /kaggle/working or saving image pixels as output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


NS3SIONNA_COMMIT = "c896b858cd9d14a1fd36a4c8116f1538b8ebbcdf"
NS3_TAG = "ns-3.40"
NS3_COMMIT = "6dceadd9f97d0b53b2af6d3b0476046fe2f09108"

# The first three image hashes come from the frozen v1 train manifest. Kodak is
# an evaluation-only probe. The added v2 sources have not had splits frozen;
# their samples are path/decode checks, not training membership.
IMAGE_SAMPLES = [
    {
        "source": "DIV2K",
        "handle": "soumikrakshit/div2k-high-resolution-images",
        "path": "DIV2K_train_HR/DIV2K_train_HR/0792.png",
        "split_scope": "official train; frozen v1 manifest",
        "expected_sha256": "62ecf6f340c79a7c6df5d9dd0103510692f47352935dbe91950ab71bbc7ea126",
    },
    {
        "source": "COCO2017",
        "handle": "awsaf49/coco-2017-dataset",
        "path": "coco2017/train2017/000000553083.jpg",
        "split_scope": "official train; frozen v1 manifest",
        "expected_sha256": "fedb9b5597d5e6e68eaa5bb03048cc9ccc7a2e6ccf9d14226352c705ddd19f69",
    },
    {
        "source": "TextOCR",
        "handle": "deeplumiere/textocr",
        "path": "textocr/textocr/textocr/train_images/58e57369b37a6901.jpg",
        "split_scope": "official train; frozen v1 manifest",
        "expected_sha256": "2f34ebbec782b9e0003718bec22c3472d0310e2a1f600466ae7a86f2ea3e07be",
    },
    {
        "source": "Kodak",
        "handle": "sherylmehta/kodak-dataset",
        "path": "kodim01.png",
        "split_scope": "evaluation-only probe; do not train",
        "expected_sha256": None,
    },
    {
        "source": "LSDIR_HR",
        "handle": "parampandya0310/hr-image-denoising-lsdir",
        "path": "shard-00/0001000/0000001.png",
        "split_scope": "source probe only; split not frozen",
        "expected_sha256": None,
    },
    {
        "source": "Flickr2K_HR",
        "handle": "hliang001/flickr2k",
        "path": "Flickr2K/Flickr2K_HR/000001.png",
        "split_scope": "source probe only; split not frozen",
        "expected_sha256": None,
    },
    {
        "source": "ImageNet_part0",
        "handle": "hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0",
        "path": "train/train-00000-of-01024",
        "split_scope": "official train shard; version/label schema to verify",
        "kind": "tfrecord",
    },
    {
        "source": "ImageNet_part1",
        "handle": "hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1",
        "path": "train/train-00500-of-01024",
        "split_scope": "official train shard; version/label schema to verify",
        "kind": "tfrecord",
    },
    {
        "source": "DocLayNet",
        "handle": "ayoubelmendoub/doclaynet",
        "path": "DocLayNet_core/COCO/train.json",
        "split_scope": "official train annotation; resolve one referenced page",
        "kind": "doclaynet_train_annotation",
    },
    {
        "source": "FFHQ_1024",
        "handle": "rahulbhalley/ffhq-1024x1024",
        "path": "images1024x1024/00000.png",
        "split_scope": "proposed train-side probe only; split not frozen",
        "expected_sha256": None,
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


class Pilot:
    def __init__(self, output: Path):
        self.output = output
        self.logs = output / "logs"
        self.output.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir()
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.results: dict[str, Any] = {"datasets": {}, "radio": {}}

    def event(self, name: str, status: str, **details: Any) -> None:
        row = {"stage": name, "status": status, "elapsed_seconds": round(time.monotonic() - self.started, 3), **details}
        self.events.append(row)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    def run(self, name: str, command: list[str], *, cwd: Path | None = None,
            env: dict[str, str] | None = None, timeout: int = 600) -> dict[str, Any]:
        log_path = self.logs / f"{name}.log"
        start = time.monotonic()
        self.event(name, "started", command=command, cwd=str(cwd) if cwd else None, log=str(log_path))
        timed_out = False
        process: subprocess.Popen[str] | None = None
        offset = 0
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                    reader.seek(offset)
                    fresh = reader.read()
                    offset = reader.tell()
                if fresh:
                    print(fresh, end="", flush=True)
                if time.monotonic() - start >= timeout:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(2)
        with log_path.open("r", encoding="utf-8", errors="replace") as reader:
            reader.seek(offset)
            tail = reader.read()
            if tail:
                print(tail, end="", flush=True)
        result = {
            "status": "timeout" if timed_out else "passed" if process and process.returncode == 0 else "failed",
            "returncode": process.returncode if process else None,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "log": str(log_path),
        }
        self.event(name, result["status"], **{k: v for k, v in result.items() if k != "status"})
        return result


def image_probe(pilot: Pilot, kagglehub: Any, sample: dict[str, Any]) -> dict[str, Any]:
    import PIL
    from PIL import Image

    name = sample["source"]
    result: dict[str, Any] = {
        "handle": sample["handle"],
        "requested_path": sample["path"],
        "split_scope": sample["split_scope"],
        "status": "failed",
    }
    try:
        resolved = Path(kagglehub.dataset_download(sample["handle"], path=sample["path"]))
        if not resolved.is_file():
            raise FileNotFoundError(f"KaggleHub returned no file: {resolved}")
        result["resolved_path"] = str(resolved)
        result["bytes"] = resolved.stat().st_size
        result["sha256"] = sha256_file(resolved)
        if sample.get("expected_sha256"):
            result["expected_sha256"] = sample["expected_sha256"]
            result["manifest_hash_match"] = result["sha256"] == sample["expected_sha256"]
        with Image.open(resolved) as image:
            image.verify()
        with Image.open(resolved) as image:
            image.load()  # verify() does not decode all pixel data.
            rgb = image.convert("RGB")
            rgb.load()
            result.update({"format": image.format, "width": image.width, "height": image.height,
                           "mode": image.mode, "decoded_rgb_shape": [rgb.height, rgb.width, 3],
                           "pillow": PIL.__version__})
        result["status"] = "passed" if result.get("manifest_hash_match", True) else "hash_mismatch"
    except Exception as exc:
        result.update({"error": str(exc), "traceback": traceback.format_exc()})
    pilot.results["datasets"][name] = result
    pilot.event("dataset_sample_" + name.lower(), result["status"], **{k: v for k, v in result.items() if k != "status"})
    return result


def doclaynet_page_probe(pilot: Pilot, kagglehub: Any) -> None:
    entry = pilot.results["datasets"].get("DocLayNet", {})
    if entry.get("status") != "passed":
        result = {"status": "skipped", "reason": "DocLayNet train annotation did not pass"}
        pilot.results["datasets"]["DocLayNet_train_page"] = result
        pilot.event("dataset_sample_doclaynet_train_page", "skipped", **result)
        return
    try:
        annotation_path = Path(entry["resolved_path"])
        with annotation_path.open("r", encoding="utf-8") as stream:
            annotation = json.load(stream)
        images = annotation.get("images", [])
        if not images:
            raise ValueError("DocLayNet train.json has no images array entries")
        file_name = str(images[0]["file_name"])
        if file_name.startswith("DocLayNet_core/"):
            page_path = file_name
        elif file_name.startswith("PNG/"):
            page_path = "DocLayNet_core/" + file_name
        else:
            page_path = "DocLayNet_core/PNG/" + Path(file_name).name
        page = {
            "handle": "ayoubelmendoub/doclaynet",
            "requested_path": page_path,
            "split_scope": "first page referenced by official train annotation",
        }
        result = image_probe(pilot, kagglehub, {**page, "source": "DocLayNet_train_page"})
        pilot.results["datasets"]["DocLayNet"]["annotation_images"] = len(images)
        pilot.results["datasets"]["DocLayNet"]["train_page_probe"] = result
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        pilot.results["datasets"]["DocLayNet_train_page"] = result
        pilot.results["datasets"].setdefault("DocLayNet", {})["train_page_probe"] = result
        pilot.event("dataset_sample_doclaynet_train_page", "failed", **result)


def build_sionna_environment(pilot: Pilot, work: Path) -> Path | None:
    env_dir = work / "sionna-venv"
    uv_python_dir = work / "uv-python"
    uv_cache_dir = work / "uv-cache"
    env = os.environ.copy()
    env.update({"UV_PYTHON_INSTALL_DIR": str(uv_python_dir), "UV_CACHE_DIR": str(uv_cache_dir)})
    setup = pilot.run("install_uv", [sys.executable, "-m", "pip", "install", "--no-cache-dir", "uv==0.13.4"], timeout=300)
    if setup["status"] != "passed":
        return None
    uv = shutil.which("uv")
    if not uv:
        scripts = Path(sys.executable).parent
        uv = str(scripts / ("uv.exe" if os.name == "nt" else "uv"))
    setup = pilot.run("install_python312", [uv, "python", "install", "3.12"], env=env, timeout=600)
    if setup["status"] != "passed":
        return None
    setup = pilot.run("create_sionna_venv", [uv, "venv", "--python", "3.12", str(env_dir)], env=env, timeout=300)
    if setup["status"] != "passed":
        return None
    python = env_dir / "bin" / "python"
    if os.name == "nt":
        python = env_dir / "Scripts" / "python.exe"
    setup = pilot.run("install_pinned_sionna_rt", [
        uv, "pip", "install", "--python", str(python),
        "sionna==1.2.1", "sionna-rt==1.2.1", "tensorflow[and-cuda]==2.19.1",
        "pyzmq", "GPUtil", "millify", "Pillow",
    ], env=env, timeout=1800)
    if setup["status"] != "passed":
        return None
    check = pilot.run("sionna_environment_check", [str(python), "-c",
        "import json,sys,sionna.rt,mitsuba as mi; "
        "assert mi.variant() == 'cuda_ad_mono_polarized', 'unexpected Mitsuba RT variant: '+str(mi.variant()); "
        "import tensorflow as tf,sionna; "
        "g=tf.config.list_physical_devices('GPU'); assert g, 'TensorFlow sees no GPU'; "
        "print(json.dumps({'python':sys.version,'tensorflow':tf.__version__,'sionna':sionna.__version__,'mitsuba':mi.__version__,'tf_gpus':[d.name for d in g]}))"],
        env={**env, "TF_CPP_MIN_LOG_LEVEL": "2"}, timeout=240)
    if check["status"] != "passed":
        return None
    return python


def decode_imagenet_records(pilot: Pilot, python: Path) -> None:
    paths = [
        pilot.results["datasets"].get("ImageNet_part0", {}).get("resolved_path"),
        pilot.results["datasets"].get("ImageNet_part1", {}).get("resolved_path"),
    ]
    if not all(paths):
        pilot.results["datasets"]["ImageNet_TFRecord_samples"] = {
            "status": "skipped", "reason": "one or both record shards were not downloaded"}
        pilot.event("imagenet_tfrecord_decode", "skipped", reason="one or both record shards were not downloaded")
        return
    code = r'''import json,sys,tensorflow as tf
out=[]
for filename in sys.argv[1:]:
    record=next(iter(tf.data.TFRecordDataset(filename).take(1)),None)
    if record is None: raise RuntimeError("empty TFRecord: "+filename)
    example=tf.train.Example.FromString(bytes(record.numpy()))
    fields=example.features.feature
    image_bytes=None
    for key in ("image/encoded","image"):
        if key in fields:
            image_bytes=fields[key].bytes_list.value[0]; break
    if image_bytes is None: raise KeyError("no encoded image field; keys="+str(list(fields)))
    image=tf.io.decode_jpeg(image_bytes,channels=3)
    label=None
    for key in ("image/class/label","label","class"):
        if key in fields:
            value=fields[key]
            label=(list(value.int64_list.value) or list(value.float_list.value) or list(value.bytes_list.value))[:1]
            break
    out.append({"file":filename,"record_bytes":int(len(record.numpy())),"decoded_shape":list(image.shape),"label":label,"feature_keys":sorted(fields.keys())})
print(json.dumps(out))
'''
    result = pilot.run("imagenet_tfrecord_decode", [str(python), "-c", code, *[str(p) for p in paths]],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"}, timeout=300)
    if result["status"] == "passed":
        try:
            text = (pilot.logs / "imagenet_tfrecord_decode.log").read_text(encoding="utf-8", errors="replace")
            start = text.rfind("[{\"file\"")
            if start >= 0:
                decoded = json.loads(text[start:].strip())
                if len(decoded) != 2 or any(not entry.get("decoded_shape") for entry in decoded):
                    raise ValueError("expected two decoded image records from ImageNet shards")
                pilot.results["datasets"]["ImageNet_TFRecord_samples"] = {"status": "passed", "records": decoded}
                for key in ("ImageNet_part0", "ImageNet_part1"):
                    if pilot.results["datasets"].get(key, {}).get("status") == "downloaded":
                        pilot.results["datasets"][key]["status"] = "passed"
            else:
                raise ValueError("decoder log did not contain the expected record JSON result")
        except Exception as exc:
            pilot.results["datasets"]["ImageNet_TFRecord_samples"] = {
                "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            pilot.event("imagenet_tfrecord_result_parse", "failed", error=str(exc))
    else:
        pilot.results["datasets"]["ImageNet_TFRecord_samples"] = {
            "status": "failed", "reason": "ImageNet TFRecord decoder process did not pass", **result}


def run_ns3_sionna(pilot: Pilot, python: Path, work: Path) -> None:
    source = work / "ns-3-dev-git"
    contrib = source / "contrib" / "sionna"
    env = os.environ.copy()
    env.update({"USER": "kaggle", "CUDA_VISIBLE_DEVICES": "0", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1"})
    apt = pilot.run("install_native_build_dependencies", ["apt-get", "update"], env=env, timeout=300)
    if apt["status"] != "passed":
        return
    apt = pilot.run("install_native_build_dependencies_packages", ["apt-get", "install", "-y",
        "g++", "cmake", "pkg-config", "libzmq3-dev", "libprotobuf-dev", "protobuf-compiler"], env=env, timeout=600)
    if apt["status"] != "passed":
        return
    clone = pilot.run("clone_ns3_340", ["git", "clone", "--depth", "1", "--branch", NS3_TAG,
        "https://github.com/nsnam/ns-3-dev-git.git", str(source)], env=env, timeout=300)
    if clone["status"] != "passed":
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != NS3_COMMIT:
        pilot.event("verify_ns3_revision", "failed", expected=NS3_COMMIT, observed=revision)
        return
    contrib.parent.mkdir(parents=True, exist_ok=True)
    clone = pilot.run("clone_ns3sionna", ["git", "clone", "--depth", "1",
        "https://github.com/tkn-tub/ns3sionna.git", str(contrib)], env=env, timeout=300)
    if clone["status"] != "passed":
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=contrib, text=True).strip()
    if revision != NS3SIONNA_COMMIT:
        pilot.event("verify_ns3sionna_revision", "failed", expected=NS3SIONNA_COMMIT, observed=revision)
        return
    configure = pilot.run("configure_ns3sionna", [sys.executable, "ns3", "configure", "--enable-examples", "--disable-tests"],
                          cwd=source, env=env, timeout=900)
    if configure["status"] != "passed":
        return
    model_dir = contrib / "model" / "ns3sionna"
    server_log = pilot.logs / "sionna_rt_server.log"
    server_env = {**env, "PYTHONPATH": str(model_dir)}
    launcher = work / "launch_sionna_server.py"
    launcher.write_text(
        "import sionna.rt, mitsuba as mi, runpy, sys\n"
        "assert mi.variant() == 'cuda_ad_mono_polarized', f'unexpected Sionna RT variant: {mi.variant()}'\n"
        f"sys.argv=['ns3sionna_server.py','--model_folder','models/','--rt_fast','--single_run','--rt_max_parallel_links','8','--verbose']\n"
        f"runpy.run_path({str(model_dir / 'ns3sionna_server.py')!r},run_name='__main__')\n",
        encoding="utf-8",
    )
    pilot.event("start_sionna_rt_server", "started", command=[str(python), str(launcher)], log=str(server_log))
    with server_log.open("w", encoding="utf-8") as stream:
        server = subprocess.Popen([str(python), str(launcher)], cwd=model_dir, env=server_env,
            stdout=stream, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 180
    ready = False
    while time.monotonic() < deadline and server.poll() is None:
        try:
            ready = "Sionna server socket ready" in server_log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        if ready:
            break
        time.sleep(2)
    if not ready:
        if server.poll() is None:
            server.terminate()
        pilot.event("sionna_rt_server_ready", "failed", returncode=server.poll(), log=str(server_log))
        return
    pilot.event("sionna_rt_server_ready", "passed", log=str(server_log))
    run_env = {**env, "PYTHONUNBUFFERED": "1"}
    example = pilot.run("ns3_sionna_rt_example", [sys.executable, "ns3", "run", "ns3sionna-example-sionna-snr"],
                        cwd=source, env=run_env, timeout=1800)
    try:
        server.wait(timeout=180)
    except subprocess.TimeoutExpired:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    trace_path = source / "snr-trace.txt"
    rows: list[list[float]] = []
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                values = [float(value) for value in line.split()]
                if len(values) >= 3 and all(math.isfinite(value) for value in values[:3]):
                    rows.append(values[:3])
            except ValueError:
                continue
    server_text = server_log.read_text(encoding="utf-8", errors="replace")
    passed = (example["status"] == "passed" and server.returncode == 0 and bool(rows)
              and "Sionna server init sucessful" in server_text)
    pilot.results["radio"] = {
        "status": "passed" if passed else "failed",
        "integration": "ns-3 3.40 + ns3sionna + Sionna-RT; one static room SNR example",
        "ns3_commit": NS3_COMMIT,
        "ns3sionna_commit": NS3SIONNA_COMMIT,
        "python": str(python),
        "server_exit_code": server.returncode,
        "snr_trace_rows": rows,
        "trace_file": str(trace_path) if trace_path.is_file() else None,
        "server_log": str(server_log),
        "scope_limit": "single simulation smoke only; not a calibrated codec-channel benchmark",
    }
    pilot.event("ns3sionna_integration_result", pilot.results["radio"]["status"],
                server_exit_code=server.returncode, trace_rows=len(rows), trace=rows[:3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/tmp/ns3sionna-dataset-pilot"))
    args = parser.parse_args()
    pilot = Pilot(args.output)
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = "/tmp/ns3sionna-kagglehub-cache"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    provenance = {
        "python": sys.version,
        "interpreter": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "work": str(args.work),
        "output": str(args.output),
        "dataset_sample_count": len(IMAGE_SAMPLES),
        "dataset_count": len({entry["source"] for entry in IMAGE_SAMPLES}),
        "source_policy": "one sample file per proposed source; no full corpus copied; no image data saved to output",
        "dataset_version_policy": "KaggleHub latest at request time; sample hashes recorded; full version/split freeze remains required before training",
    }
    try:
        dependency_result = pilot.run("install_kagglehub", [sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "kagglehub==1.0.3", "Pillow"], timeout=420)
        if dependency_result["status"] != "passed":
            raise RuntimeError("KaggleHub/Pillow setup failed; see logs/install_kagglehub.log")
        import kagglehub
        try:
            importlib.metadata.version("kagglehub")
            provenance["kagglehub"] = importlib.metadata.version("kagglehub")
        except importlib.metadata.PackageNotFoundError:
            provenance["kagglehub"] = "imported; version unavailable"
        import torch
        provenance.update({"torch": torch.__version__,
                           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []})
        write_json(pilot.output / "environment.json", provenance)
        pilot.event("environment", "passed", **provenance)
        for sample in IMAGE_SAMPLES:
            if sample.get("kind") == "tfrecord":
                try:
                    resolved = Path(kagglehub.dataset_download(sample["handle"], path=sample["path"]))
                    if not resolved.is_file():
                        raise FileNotFoundError(str(resolved))
                    result = {"status": "downloaded", "handle": sample["handle"], "requested_path": sample["path"],
                              "resolved_path": str(resolved), "bytes": resolved.stat().st_size,
                              "sha256": sha256_file(resolved), "split_scope": sample["split_scope"]}
                    pilot.results["datasets"][sample["source"]] = result
                    pilot.event("dataset_sample_" + sample["source"].lower(), "downloaded", **{k: v for k, v in result.items() if k != "status"})
                except Exception as exc:
                    pilot.results["datasets"][sample["source"]] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                    pilot.event("dataset_sample_" + sample["source"].lower(), "failed", error=str(exc))
            elif sample.get("kind") == "doclaynet_train_annotation":
                try:
                    resolved = Path(kagglehub.dataset_download(sample["handle"], path=sample["path"]))
                    if not resolved.is_file():
                        raise FileNotFoundError(str(resolved))
                    result = {"status": "passed", "handle": sample["handle"], "requested_path": sample["path"],
                              "resolved_path": str(resolved), "bytes": resolved.stat().st_size,
                              "sha256": sha256_file(resolved), "split_scope": sample["split_scope"]}
                    pilot.results["datasets"][sample["source"]] = result
                    pilot.event("dataset_sample_doclaynet_train_annotation", "passed", **{k: v for k, v in result.items() if k != "status"})
                except Exception as exc:
                    pilot.results["datasets"][sample["source"]] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                    pilot.event("dataset_sample_doclaynet_train_annotation", "failed", error=str(exc))
            else:
                image_probe(pilot, kagglehub, sample)
        doclaynet_page_probe(pilot, kagglehub)
    except Exception as exc:
        pilot.event("dataset_probe_setup", "failed", error=str(exc), traceback=traceback.format_exc())

    sionna_python = None
    try:
        sionna_python = build_sionna_environment(pilot, args.work)
        if sionna_python:
            decode_imagenet_records(pilot, sionna_python)
            run_ns3_sionna(pilot, sionna_python, args.work)
        else:
            pilot.event("ns3sionna_setup", "failed", reason="Sionna RT environment was not built; see stage logs")
    except Exception as exc:
        pilot.event("ns3sionna_setup", "failed", error=str(exc), traceback=traceback.format_exc())

    required_dataset_checks = [
        "DIV2K", "COCO2017", "TextOCR", "Kodak", "LSDIR_HR", "Flickr2K_HR",
        "ImageNet_part0", "ImageNet_part1", "ImageNet_TFRecord_samples",
        "DocLayNet", "DocLayNet_train_page", "FFHQ_1024",
    ]
    dataset_results = list(pilot.results["datasets"].values())
    required_status = {name: pilot.results["datasets"].get(name, {}).get("status", "missing")
                       for name in required_dataset_checks}
    data_passed = all(status == "passed" for status in required_status.values())
    image_net_decoded = required_status["ImageNet_TFRecord_samples"] == "passed"
    radio_passed = pilot.results.get("radio", {}).get("status") == "passed"
    status = "complete" if data_passed and radio_passed else "partial_or_failed"
    final = {
        "status": status,
        "elapsed_seconds": round(time.monotonic() - pilot.started, 3),
        "dataset_source_results": pilot.results["datasets"],
        "required_dataset_check_status": required_status,
        "datasets_passed": sum(item.get("status") == "passed" for item in dataset_results),
        "datasets_total": len(dataset_results),
        "imagenet_record_decode_passed": image_net_decoded,
        "radio": pilot.results.get("radio", {}),
        "paper_training_or_codec_evaluation_started": False,
        "full_dataset_copy_performed": False,
        "pixels_saved_as_output": False,
    }
    write_json(pilot.output / "status.json", final)
    write_json(pilot.output / "results.json", pilot.results)
    pilot.event("final", status, datasets_passed=final["datasets_passed"], datasets_total=final["datasets_total"],
                radio_status=pilot.results.get("radio", {}).get("status", "not_run"),
                elapsed_seconds=final["elapsed_seconds"])
    print(json.dumps({"pilot_status": status, "output": str(pilot.output),
                      "dataset_count": final["datasets_total"], "radio_status": pilot.results.get("radio", {}).get("status", "not_run")}), flush=True)


if __name__ == "__main__":
    main()
