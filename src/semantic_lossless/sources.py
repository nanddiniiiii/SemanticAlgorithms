"""Fetch immutable author sources; keep checkpoints out of source downloads."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import uuid
import zipfile
from .artifacts import write_json
from .paths import project_root

SOURCES = {
    "swinjscc": ("semcomm/SwinJSCC", "a6d0e6da53548976acbe9317839a077ef31f190f"),
    "jcm": ("SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication", "9a53ef0a3df2f41ecae6cd531968f70a13d53088"),
    "shared_semantic_codebook": ("SJTU-mxtao/semantic-communication-w-codebook", "ee4ea1d3395a5100d79ebe7e19109a0c99d2eec6"),
    "sqgan": ("frapez1/SQ-GAN", "0e5cdd7c0240b89749dad684645c53f783ac24cc"),
    "mdjcm": ("zhang-guangyi/Semantic-MDJCM", "a52b13dbbac0873b45a450074d73106bae61ffb0"),
    "mambajscc": ("Wireless3C-SJTU/MambaJSCC", "8323f9799f48c0efddce0521100d5c4e12ebf699"),
    "diffcom": ("wsxtyrdd/diffcom", "a8cc4d63304f5bb514d69576e689ff2412510161"),
    "diffjscc": ("mingyuyng/DiffJSCC", "13aeb62451b872ce41ceba132c9c30a9ca172c53"),
    "deepjscc_q": ("r0ymanesco/deepjscc_stream", "da863de5e3acb4fbb307c0cf8dfe830303e99aa7"),
    "amp_sc": ("gmzSZU/AMP-SC", "063d8098633cf31acd236982b8510d969d225d7c"),
}
JCM_FILES = ["network.py", "modules.py", "analog_network.py", "main.py", "train.py",
             "train_analog.py", "evaluation.py", "utils.py", "requirements.txt", "README.md"]


def _download(url, maximum=128 * 1024 * 1024):
    request = urllib.request.Request(url, headers={"User-Agent": "SemanticLossless/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError("Source download exceeded its size limit; weights must use the weight downloader")
    return content


def source_path(name):
    return project_root() / ".upstream" / name


def fetch_source(name):
    if name not in SOURCES:
        raise ValueError(f"No pinned upstream fetch configured for {name}")
    repository, commit = SOURCES[name]
    target = source_path(name)
    receipt = target / "source_receipt.json"
    if target.exists():
        if not receipt.is_file():
            raise FileExistsError(f"Existing source directory lacks a receipt: {target}")
        record = json.loads(receipt.read_text())
        if record["repository"] != repository or record["commit"] != commit:
            raise ValueError("Existing source belongs to a different revision")
        for item in record["files"]:
            if hashlib.sha256((target / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Upstream source was modified: {item['path']}")
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{name}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        if name == "jcm":
            # Its repository includes gigabytes of checkpoints. Fetch source only.
            for filename in JCM_FILES:
                content = _download(f"https://raw.githubusercontent.com/{repository}/{commit}/{filename}")
                (temporary / filename).write_bytes(content)
        else:
            content = _download(f"https://codeload.github.com/{repository}/zip/{commit}")
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(member.file_size for member in archive.infolist()) > 256 * 1024 * 1024:
                    raise ValueError("Expanded source archive exceeds size limit")
                for member in archive.infolist():
                    parts = Path(member.filename.replace("\\", "/")).parts[1:]
                    if not parts or member.is_dir():
                        continue
                    destination = temporary.joinpath(*parts)
                    destination.resolve().relative_to(temporary.resolve())
                    if stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError("Upstream archive contains a symbolic link")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(member))
        files = [{"path": path.relative_to(temporary).as_posix(),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(temporary.rglob("*")) if path.is_file()]
        record = {"repository": repository, "commit": commit, "files": files,
                  "source_fetched": True, "execution_verified": False}
        write_json(temporary / "source_receipt.json", record)
        temporary.rename(target)
        return record
    finally:
        if temporary.exists():
            # Only this invocation's uniquely named extraction directory.
            shutil.rmtree(temporary)
