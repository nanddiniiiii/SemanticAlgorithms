"""Build a private Kaggle pilot for ns-3/Sionna RT plus bounded dataset reads."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ns3sionna_dataset_pilot.py"
SAMPLE_FAMILIES = [
    "DIV2K", "COCO2017", "TextOCR", "Kodak", "LSDIR_HR", "Flickr2K_HR",
    "ImageNet-1k original-resolution TFRecords (both parts)", "DocLayNet", "FFHQ_1024",
]

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("sources.lock.json", (ROOT / "sources.lock.json").read_bytes())
    archive.write(RUNNER, "scripts/run_ns3sionna_dataset_pilot.py")
payload = buffer.getvalue()
bundle_sha = hashlib.sha256(payload).hexdigest()
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
script = f'''import base64,hashlib,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path
bundle=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(bundle).hexdigest()=={bundle_sha!r}
root=Path('/kaggle/working/Semantic_Lossless');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    for name in archive.namelist():(root/name).resolve().relative_to(root.resolve())
    archive.extractall(root)
os.chdir(root);os.environ['SEMANTIC_LOSSLESS_ROOT']=str(root)
out=root/'Results/stage1_operations/ns3sionna_dataset_pilot/v1'
provenance={{'source_bundle_sha256':hashlib.sha256(bundle).hexdigest(),'root':str(root),
 'repository_revision':{revision!r},'working_tree_dirty':{dirty!r},
 'purpose':'one static ns-3/Sionna-RT integration smoke plus one-file decode checks for all proposed custom-data sources',
 'dataset_source_families':{SAMPLE_FAMILIES!r},'kaggle_dataset_sources_attached':[],
 'data_access':'KaggleHub single-file path requests; no full corpus staging/copy',
 'output':str(out),'python':sys.version,'interpreter':sys.executable}}
(root/'source_bundle.json').write_text(json.dumps(provenance,indent=2)+'\\n')
print(json.dumps({{'stage':'source_preflight','status':'passed',**provenance}}),flush=True)
logs=root/'run_logs';logs.mkdir(exist_ok=True)
def run_logged(stage,command,timeout):
 start=time.monotonic();path=logs/(stage+'.log')
 print(json.dumps({{'stage':stage,'status':'started','command':command,'log':str(path),'bundle_sha256':{bundle_sha!r}}}),flush=True)
 offset=0
 with path.open('w',encoding='utf-8') as stream:
  process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
  timed_out=False
  while process.poll() is None:
   with path.open(encoding='utf-8',errors='replace') as reader:
    reader.seek(offset);text=reader.read();offset=reader.tell()
   if text:print(text,end='',flush=True)
   if time.monotonic()-start>timeout:
    timed_out=True;process.terminate()
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    break
   time.sleep(2)
 with path.open(encoding='utf-8',errors='replace') as reader:
  reader.seek(offset);text=reader.read()
  if text:print(text,end='',flush=True)
 result={{'stage':stage,'status':'timeout' if timed_out else 'completed' if process.returncode==0 else 'failed',
  'returncode':process.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'log':str(path)}}
 (logs/'stages.jsonl').open('a').write(json.dumps(result)+'\\n')
 print(json.dumps(result),flush=True)
 if timed_out or process.returncode:raise RuntimeError('Stage failed: '+stage+'; see '+str(path))
run_logged('ns3sionna_and_dataset_pilot',[sys.executable,'scripts/run_ns3sionna_dataset_pilot.py',
 '--output',str(out),'--work','/tmp/ns3sionna-dataset-pilot'],10500)
files=[]
for path in sorted(out.rglob('*')):
 if path.is_file():files.append({{'path':path.relative_to(out).as_posix(),'bytes':path.stat().st_size,
  'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size<50_000_000 else 'recorded-in-output-not-rehashed'}})
(out/'artifact_manifest.json').write_text(json.dumps({{'files':files}},indent=2)+'\\n')
print(json.dumps({{'stage':'artifact_manifest','status':'completed','files':len(files),'output':str(out)}}),flush=True)
'''
compile(script, "ns3sionna_dataset_pilot", "exec")
folder = ROOT / "Kaggle" / "ns3sionna_dataset_pilot"
folder.mkdir(parents=True, exist_ok=True)
(folder / "kernel.py").write_text(script, encoding="utf-8")
metadata = {
    "id": "satyapaladuguplus/ns3sionna-custom-data-pilot",
    "title": "ns-3 Sionna and Custom Dataset Pilot",
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "docker_image_pinning_type": "original",
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
receipt = {
    "kernel": metadata["id"],
    "kernel_sha256": hashlib.sha256((folder / "kernel.py").read_bytes()).hexdigest(),
    "source_bundle_sha256": bundle_sha,
    "bundle_bytes": len(payload),
    "repository_revision": revision,
    "working_tree_dirty": dirty,
    "dataset_families": SAMPLE_FAMILIES,
    "dataset_transfer_policy": "one file per source via KaggleHub; no full datasets attached or copied",
    "ns3_tag": "ns-3.40",
    "ns3_commit": "6dceadd9f97d0b53b2af6d3b0476046fe2f09108",
    "ns3sionna_commit": "c896b858cd9d14a1fd36a4c8116f1538b8ebbcdf",
}
(folder / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
