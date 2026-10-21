"""CLI for verified upstream adapters; preparation never means measured completion."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["DeepJSCC","SwinJSCC","NTSCC_plus","ADJSCC","DeepJSCC_Q","SQGAN"], required=True)
    parser.add_argument("--stage", choices=["main","semantic","image","finetune"], default="main")
    parser.add_argument("--mode", choices=["preflight","train","evaluate"], required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--kodak-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--channel", choices=["awgn","rayleigh"], default="awgn")
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    parser.add_argument("--rate-cbr",type=float,help="Realizable author bottleneck CBR for a separate fresh operating-point run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model":args.model,"stage":args.stage,"mode":args.mode,"status":"PREFLIGHT_RUNNING",
              "channel":args.channel,"training_complete":False,"evaluation_complete":False,
              "eligible_measured":False,"source_model_edits":[],"blockers":[]}
    try:
        import torch
        from .sources import verified_source
        from .adapters import TorchAdapter, required_assets
        from .data import image_paths, dataset_identity
        from .engine import seed_all, write_json, train
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        settings=contract["training"]
        early=settings.get("early_stopping",{})
        if (settings.get("unique_source_images_per_epoch")!=800 or settings.get("global_batch_size")!=16 or
            early.get("minimum_epochs")!=50 or early.get("maximum_epochs")!=150 or
            early.get("patience")!=20 or early.get("relative_min_delta")!=.001):
            raise ValueError("Contract must specify approved full800/globalbatch16/min50/max150/patience20/relative0.001 settings")
        if args.chunk_seconds<0 or (args.rate_cbr is not None and args.rate_cbr<=0):
            raise ValueError("Chunk duration must be nonnegative and requested CBR positive")
        source, manifest = verified_source(args.model)
        report.update(source_commit=manifest["commit"], source_integrity_verified=True,
                      provenance=manifest["provenance"], cuda_available=torch.cuda.is_available(),
                      cuda_devices=torch.cuda.device_count(), torch_version=torch.__version__,
                      transport="explicit_matched_channel_extension", receiver_csi="perfect" if args.channel=="rayleigh" else "not_applicable",
                      fading_scope="one complex coefficient per image/codeword" if args.channel=="rayleigh" else "not_applicable")
        report["blockers"].extend(required_assets(args.model,args.stage,contract,args.mode))
        if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count()<2):
            report["blockers"].append("Two CUDA GPUs required for approved training path")
        if args.model != "SQGAN" and args.stage != "main":
            report["blockers"].append("Only SQGAN has semantic/image/finetune stages")
        train_paths = val_paths = None
        if args.mode in {"preflight","train","evaluate"}:
            try:
                train_paths = image_paths(args.train_root,800)
                val_paths = image_paths(args.val_root,100)
                report["dataset_sha256"] = dataset_identity(train_paths,val_paths)
                args.dataset_sha256=report["dataset_sha256"]
            except (ValueError,OSError) as exc:
                report["blockers"].append(str(exc))
        if args.mode in {"preflight","evaluate"}:
            try:
                image_paths(args.kodak_root,24)
            except (ValueError,OSError) as exc:
                report["blockers"].append(str(exc))
        # Still inspect actual core compatibility on CPU, even when data/GPU absent.
        if args.model in {"DeepJSCC","SwinJSCC","DeepJSCC_Q"}:
            seed_all(contract["training"].get("seed",20261020))
            adapter = TorchAdapter(args.model,source,args.channel,
                contract["training"].get("snr_db",10),device=args.device if torch.cuda.is_available() else "cpu",
                rate_cbr=args.rate_cbr or contract["evaluation"].get("primary_cbr_budget",1/6))
            report.update(parameter_count=sum(p.numel() for p in adapter.parameters()),
                          loss=adapter.loss_name, training_crop=adapter.crop,
                          preprocessing="author ToTensor+Resize(128,128)" if args.model=="DeepJSCC" else "author RGB random 256 crop; fixed center crop validation",
                          architecture_instantiated=True)
        elif args.rate_cbr is not None:
            report["blockers"].append("Explicit fixed-rate bottleneck sweep currently implemented only for DeepJSCC/SwinJSCC/DeepJSCC_Q")
        elif args.model=="NTSCC_plus" and not report["blockers"]:
            from .ntscc import NTSCCAdapter
            adapter=NTSCCAdapter(source,args.channel,contract["training"].get("snr_db",10),args.device)
        if report["blockers"]:
            report["status"] = "BLOCKED"
            write_json(args.output/"preflight.json",report)
            print(json.dumps(report,indent=2))
            return 2
        if args.device=="cpu":
            report["status"]="PREPARATION_PASSED_RUNTIME_UNVERIFIED"
            report["blockers"].append("CPU preparation cannot admit benchmark training; actual two-GPU runtime unverified")
            write_json(args.output/"preflight.json",report)
            print(json.dumps(report,indent=2))
            return 2
        if args.model=="ADJSCC":
            from .adjscc_tf import diagnostic
            report.update(diagnostic(args.output,args.channel))
        else:
            from .diagnostics import torch_diagnostic
            report.update(torch_diagnostic(adapter,args.output))
        report["status"] = "RUNTIME_PREFLIGHT_PASSED_NOT_TRAINED"
        write_json(args.output/"preflight.json",report)
        if args.mode == "train":
            if args.model=="ADJSCC":
                from .adjscc_tf import train as tf_train
                report.update(tf_train(args,contract,manifest,train_paths,val_paths,report["dataset_sha256"]))
            else:
                report.update(train(args,contract,adapter,manifest,train_paths,val_paths,report["dataset_sha256"]))
        elif args.mode == "evaluate":
            if args.model=="ADJSCC":
                from .adjscc_tf import evaluate as tf_evaluate
                report.update(tf_evaluate(args,contract,manifest,image_paths(args.kodak_root,24)))
            else:
                from .evaluation import evaluate
                report.update(evaluate(args,contract,adapter,manifest))
        print(json.dumps(report,indent=2))
        write_json(args.output/"runtime_result.json",report)
        return 0
    except Exception as exc:
        report.update(status="BLOCKED" if isinstance(exc,(ImportError,ModuleNotFoundError)) else "FAILED",error_type=type(exc).__name__,error=str(exc))
        (args.output/"runtime_result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
