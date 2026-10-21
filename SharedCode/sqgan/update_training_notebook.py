"""Embed the reviewed session orchestration into the existing SQ-GAN notebook."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOK = ROOT / "Notebooks/project_notebooks/Training/sqgan_div2k_channel/sqgan_div2k_awgn_rayleigh_training.ipynb"
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

source = "".join(nb["cells"][1]["source"])
source = source.replace('STOP_AFTER_STAGE_EPOCHS = 50  # raise to 100, then 150 when resuming later chunks', 'STOP_AFTER_STAGE_EPOCHS = 150  # preserve the total stage target; session deadline handles chunking')
source = source.replace('REPO = Path("/kaggle/working/SQ-GAN")', 'REPO = Path("/tmp/SQ-GAN")  # source, top-k checkpoints and images stay outside exported output')
if "SESSION_HOURS" not in source:
    source = source.replace('EPOCHS_PER_STAGE = 150\n', 'EPOCHS_PER_STAGE = 150\nSESSION_HOURS = 9.5\nSESSION_DEADLINE_UNIX = time.time() + SESSION_HOURS * 3600\nos.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # legacy author torch.load behavior\n')
nb["cells"][1]["source"] = source.splitlines(keepends=True)

source = "".join(nb["cells"][3]["source"])
source = source.split("# BEGIN SESSION ORCHESTRATION")[0]
source += '\n# BEGIN SESSION ORCHESTRATION\nCOMPAT_DIR = Path("/tmp/sqgan_runtime")\nCOMPAT_DIR.mkdir(parents=True, exist_ok=True)\n'
for name in ["session_control.py", "check_resume_contract.py"]:
    source += f'(COMPAT_DIR / {name!r}).write_text({(HERE/name).read_text(encoding="utf-8")!r}, encoding="utf-8")\n'
source += '''
# Restore only the exported training state; the official source was freshly cloned above.
for input_root in Path("/kaggle/input").iterdir():
    for prior_out in input_root.rglob("sqgan_div2k_channel_results"):
        if prior_out.is_dir() and (prior_out / "stage_state").is_dir():
            shutil.copytree(prior_out, OUTPUT, dirs_exist_ok=True)
            print("RESTORED SAVED OUTPUT", prior_out, flush=True)
    for prior_logs in input_root.rglob("upstream_logs"):
        if prior_logs.is_dir():
            shutil.copytree(prior_logs, REPO / "logs", dirs_exist_ok=True)
            print("RESTORED UPSTREAM LOGS", prior_logs, flush=True)
'''
nb["cells"][3]["source"] = source.splitlines(keepends=True)

source = "".join(nb["cells"][5]["source"])
source = source.split("# BEGIN RESUME CONTRACT CHECK")[0]
source += '''
# BEGIN RESUME CONTRACT CHECK
# Exercise a real two-optimizer checkpoint/restore on a tiny CPU-only test model.
probe_env = os.environ.copy()
probe_env["PYTHONPATH"] = str(COMPAT_DIR) + os.pathsep + probe_env.get("PYTHONPATH", "")
subprocess.run([sys.executable, str(COMPAT_DIR/"check_resume_contract.py"), str(OUTPUT/"preflight_resume_contract")], env=probe_env, check=True)
'''
nb["cells"][5]["source"] = source.splitlines(keepends=True)

nb["cells"][6]["source"] = ["""## Training with verified continuation and a session deadline

The official source remains pinned and unmodified. DIV2K pseudo-segmentation and the noisy channel are explicit adaptations. Each author stage retains its 150-epoch target and original optimizer/scheduler setup. The three stages run independently for AWGN and Rayleigh.

A runtime callback verifies restored optimizer learning rates and global step, writes epoch audit records, and requests a normal stop before the 9.5-hour notebook budget ends. It does not alter the model forward path, gradients, losses, or optimizer updates. A CPU-only two-optimizer test must first demonstrate a successful stop/checkpoint/restore.

The latest complete Lightning checkpoint and configs from each stage are mirrored atomically into the output folder. Repository files, top-k duplicates and diagnostic images stay in `/tmp`. Successful output from the preceding chunk must be attached as a notebook input. Completed stages are skipped; partial stages resume through the author's `--resume` path. Evaluation runs only after all six stages are complete and enough session time remains.
"""]

old = "".join(nb["cells"][8]["source"])
boundary = old.index("    # Fail fast") if "    # Fail fast" in old else old.index("    import gc")
prefix = old[:boundary]
prefix = prefix.replace('*/checkpoints/last.ckpt', '*/checkpoints/last*.ckpt')
prefix = prefix.replace('return max(candidates,key=lambda p:p.stat().st_mtime)', 'return max(candidates,key=lambda p:(p.stat().st_mtime_ns,p.name=="last.ckpt"))')
body = '''    import gc
    import threading

    def checkpoint_info(path):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        progress = raw.get("loops", {}).get("fit_loop", {}).get("epoch_progress", {})
        result = {
            "saved_epoch_index": int(raw["epoch"]),
            # Lightning restores its completed counter one behind after a checkpoint
            # taken before epoch-end bookkeeping. Processed counts full epoch passes.
            "completed_epochs": max(int(progress.get("total", {}).get("completed", 0)), int(progress.get("total", {}).get("processed", 0)), int(raw["epoch"])),
            "global_step": int(raw["global_step"]),
            "optimizer_states": len(raw.get("optimizer_states", [])),
            "lr_schedulers": len(raw.get("lr_schedulers", [])),
            "optimizer_learning_rates": [[g["lr"] for g in opt["param_groups"]] for opt in raw.get("optimizer_states", [])],
            "state_dict_tensors": len(raw["state_dict"]),
        }
        assert result["optimizer_states"] == result["lr_schedulers"] == 2, result
        del raw
        gc.collect()
        return result

    for split_name, image_root, label_root in (("train",DIV2K_TRAIN,train_labels),("validation",DIV2K_VAL,val_labels),("kodak",KODAK_ROOT,kodak_labels)):
        images = [p for p in Path(image_root).rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg"}]
        assert images and all((Path(label_root)/(p.stem+".png")).is_file() for p in images), split_name
        print("DATASET CHECK PASS", split_name, len(images), flush=True)

    compat_dir = COMPAT_DIR
    (compat_dir / "sitecustomize.py").write_text(SHIM_SOURCE, encoding="utf-8")
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(compat_dir) + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    child_env["SQGAN_STOP_AFTER_STAGE_EPOCHS"] = str(STOP_AFTER_STAGE_EPOCHS)
    child_env["SQGAN_SESSION_DEADLINE_UNIX"] = str(SESSION_DEADLINE_UNIX)
    child_env["SQGAN_OUTPUT_DIR"] = str(OUTPUT)

    source_diff = subprocess.run(["git","diff","--exit-code",UPSTREAM_COMMIT,"--"],cwd=REPO,capture_output=True,text=True)
    assert source_diff.returncode == 0, source_diff.stdout + source_diff.stderr
    assert subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip() == UPSTREAM_COMMIT
    print("UPSTREAM SOURCE INTEGRITY PASS", UPSTREAM_COMMIT, flush=True)
    # Release all GPU tensors from pseudo-labelling before the distributed training subprocess.
    del segmenter, processor
    gc.collect()
    torch.cuda.empty_cache()

    mirror_stop = threading.Event()
    mirror_lock = threading.Lock()
    mirrored_signatures = {}
    def mirror_checkpoints(force=False):
        logs_root = REPO / "logs"
        if not logs_root.exists(): return
        # Lightning may use last-v1.ckpt after restoring into a different log directory.
        # Export only the newest variant per stage under the canonical last.ckpt name.
        newest = {}
        for candidate in logs_root.glob("*sqgan_*_150ep/checkpoints/last*.ckpt"):
            key = candidate.parent
            if key not in newest or (candidate.stat().st_mtime_ns, candidate.name == "last.ckpt") > (newest[key].stat().st_mtime_ns, newest[key].name == "last.ckpt"):
                newest[key] = candidate
        candidates = list(newest.values()) + list(logs_root.glob("*sqgan_*_150ep/configs/*.yaml"))
        for src in candidates:
            try:
                stat = src.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                previous = mirrored_signatures.get(str(src))
                mirrored_signatures[str(src)] = signature
                if not force and previous != signature: continue
                relative = src.relative_to(logs_root)
                if src.name.startswith("last") and src.suffix == ".ckpt": relative = relative.with_name("last.ckpt")
                dst = OUTPUT / "upstream_logs" / relative
                if dst.exists() and dst.stat().st_size == stat.st_size and dst.stat().st_mtime_ns >= stat.st_mtime_ns: continue
                dst.parent.mkdir(parents=True,exist_ok=True)
                temporary = dst.with_name(dst.name+".partial")
                shutil.copy2(src,temporary)
                after = src.stat()
                if (after.st_size,after.st_mtime_ns) == signature and temporary.stat().st_size == stat.st_size:
                    os.replace(temporary,dst)
                    print("CHECKPOINT EXPORTED", str(dst.relative_to(OUTPUT)), stat.st_size, flush=True)
            except (OSError,FileNotFoundError) as exc:
                print("CHECKPOINT EXPORT RETRY", src.name, repr(exc), flush=True)

    _mirror_without_lock = mirror_checkpoints
    def mirror_checkpoints(force=False):
        with mirror_lock:
            _mirror_without_lock(force)

    def mirror_loop():
        while not mirror_stop.is_set():
            mirror_checkpoints()
            mirror_stop.wait(20)

    mirror_thread = threading.Thread(target=mirror_loop,daemon=True)
    mirror_thread.start()
    stage_state_dir = OUTPUT / "stage_state"
    stage_state_dir.mkdir(exist_ok=True)
    chunk_paused = False
    try:
        for channel in CHANNELS:
            channel_out = OUTPUT / channel
            channel_out.mkdir(exist_ok=True)
            parts = channel_out / "components"
            for mode in ("ssm","img","all"):
                marker = stage_state_dir / f"{channel}_{mode}.json"
                if marker.is_file() and json.loads(marker.read_text()).get("status") == "complete":
                    print("STAGE ALREADY COMPLETE",channel,mode,flush=True)
                    continue
                if time.time()+1200 >= SESSION_DEADLINE_UNIX:
                    chunk_paused=True
                    break
                cfg=channel_config(channel,mode,parts if mode=="all" else None)
                cfg_path=OUTPUT/f"config_{channel}_{mode}.yaml"
                OmegaConf.save(cfg,cfg_path)
                args=[sys.executable,"train.py","--base",str(cfg_path),"--mode",mode,"--max_epochs",str(EPOCHS_PER_STAGE),"--gpus","0,1","--precision","32","--logtype","tensorboard","--check_val_every_n_epoch","1","--num_sanity_val_steps","0"]
                args += ["--enable_progress_bar","false"]  # compact measured progress comes from SessionControl
                candidates=list((REPO/"logs").glob(f"*sqgan_{channel}_{mode}_150ep/checkpoints/last.ckpt"))
                stage_env=child_env.copy()
                stage_env["SQGAN_STAGE_AUDIT_DIR"]=str(OUTPUT/"audit"/f"{channel}_{mode}")
                if candidates:
                    resume=max(candidates,key=lambda p:p.stat().st_mtime)
                    expected=checkpoint_info(resume)
                    stage_env["SQGAN_EXPECTED_RESUME"]=json.dumps(expected)
                    args += ["--resume",str(resume.parents[1])]
                    print("RESUMING AUTHOR CHECKPOINT",channel,mode,json.dumps(expected),flush=True)
                else:
                    stage_env["SQGAN_EXPECTED_RESUME"]="null"
                    args += ["--name",f"sqgan_{channel}_{mode}_150ep"]
                if mode=="all": args += ["--ckpt_path_parts",str(parts)]
                print("TRAIN",channel,mode," ".join(args),flush=True)
                subprocess.run(args,cwd=REPO,env=stage_env,check=True)
                ckpt=latest_last_ckpt(channel,mode)
                info=checkpoint_info(ckpt)
                mirror_checkpoints(force=True)
                exported=OUTPUT/"upstream_logs"/ckpt.relative_to(REPO/"logs").with_name("last.ckpt")
                assert exported.is_file() and exported.stat().st_size==ckpt.stat().st_size
                info.update({"channel":channel,"mode":mode,"target_epochs":EPOCHS_PER_STAGE,"checkpoint_relative":str(exported.relative_to(OUTPUT)),"bytes":exported.stat().st_size,"status":"complete" if info["completed_epochs"]>=EPOCHS_PER_STAGE else "partial","updated_unix":time.time()})
                marker.write_text(json.dumps(info,indent=2))
                print("SAVED STAGE STATE",json.dumps(info),flush=True)
                if info["status"] != "complete":
                    chunk_paused=True
                    break
                if mode=="ssm": split_trained_components(ckpt,parts,["encoder_semantic","decoder_semantic","masker_semantic","demasker_semantic","quantize_semantic"])
                elif mode=="img": split_trained_components(ckpt,parts,["encoder","decoder","masker","demasker","quantize"])
                else: (channel_out/"final_checkpoint.json").write_text(json.dumps(info,indent=2))
            if chunk_paused: break
    finally:
        mirror_stop.set()
        mirror_thread.join(timeout=120)
        mirror_checkpoints(force=True)
    states=[json.loads(p.read_text()) for p in stage_state_dir.glob("*.json")]
    training_complete=len(states)==6 and all(p.get("status")=="complete" and p.get("completed_epochs",0)>=EPOCHS_PER_STAGE for p in states)
    manifest={"upstream_commit":UPSTREAM_COMMIT,"model":"SQ-GAN-DIV2K-channel-adaptation","session_budget_hours":SESSION_HOURS,"channels":CHANNELS,"snr_db":SNR_DB,"epochs_per_stage":EPOCHS_PER_STAGE,"stages":states,"training_complete":training_complete,"comparison_ready":False,"exported_files":[{"path":str(p.relative_to(OUTPUT)),"bytes":p.stat().st_size} for p in OUTPUT.rglob("*") if p.is_file()]}
    (OUTPUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("ALL TRAINING COMPLETE" if training_complete else "CHUNK SAVED NORMALLY; CONTINUATION REQUIRED",json.dumps(states),flush=True)
'''
shim = '''import os, sys
import numpy as np
if not hasattr(np, "Inf"): np.Inf=np.inf
sys.path.insert(0,os.getcwd())
import pytorch_lightning.plugins as _plugins
from pytorch_lightning.strategies import DDPStrategy
_plugins.DDPPlugin=DDPStrategy
from utils.logger import SetupCallback, CaptionImageLogger
SetupCallback.on_fit_start=SetupCallback.on_pretrain_routine_start
delattr(SetupCallback,"on_pretrain_routine_start")
_upstream_tensorboard=CaptionImageLogger._tensorboard
def _tensorboard(self,pl_module,images,batch_idx,split):
    safe={k:v for k,v in images.items() if k not in ("ordered_vqidk","semantic_ordered_vqidk")}
    return _upstream_tensorboard(self,pl_module,safe,batch_idx,split)
CaptionImageLogger._tensorboard=_tensorboard
from session_control import install
install()
'''
source=prefix+body.replace("SHIM_SOURCE",repr(shim))
nb["cells"][8]["source"]=source.splitlines(keepends=True)

source="".join(nb["cells"][10]["source"])
source=source.replace('if RUN_MODE == "full" and training_complete:', 'if RUN_MODE == "full" and training_complete and time.time()+1800 < SESSION_DEADLINE_UNIX:')
source=source.replace('final_ckpt=channel_out/"final_last.ckpt"', 'final_ckpt=OUTPUT/json.loads((channel_out/"final_checkpoint.json").read_text())["checkpoint_relative"]')
source=source.replace('shutil.copy2(final_ckpt,OUTPUT/f"SQGAN_{channel}_final_last.ckpt")', 'torch.save(eval_model.state_dict(),OUTPUT/f"SQGAN_{channel}_model.pth")')
source=source.replace('eval_model.train(); train_batch=', '''eval_model.train()
        # Match the official final-stage freeze when profiling training forward/backward.
        for name in ("encoder_semantic","decoder_semantic","masker_semantic","demasker_semantic","quantize_semantic"):
            for parameter in getattr(eval_model,name).parameters(): parameter.requires_grad=False
        train_batch=''')
if 'manifest["comparison_ready"]=True' not in source:
    source += '\n    manifest["comparison_ready"]=True\n    manifest["evaluation_complete"]=True\n    (OUTPUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2))\n'
nb["cells"][10]["source"]=source.splitlines(keepends=True)

for index,cell in enumerate(nb["cells"]):
    if cell["cell_type"]=="code":
        ast.parse("".join(cell["source"]),filename=f"cell_{index}")
compile(shim,"sitecustomize.py","exec")
NOTEBOOK.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
metadata_path=NOTEBOOK.parent/"kernel-metadata.json"
metadata=json.loads(metadata_path.read_text())
metadata["machine_shape"]="NvidiaTeslaT4"  # Kaggle's CLI value maps to the verified 2xT4 machine
metadata["kernel_sources"]=["jagan028/div2k-dataset-generation-for-isr","hmmkk20/sqgan-checkpoint-recovery-20261020"]
metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
print("Updated notebook orchestration and recovery input:",NOTEBOOK)
