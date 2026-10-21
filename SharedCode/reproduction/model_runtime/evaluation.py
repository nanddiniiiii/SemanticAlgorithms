"""Native full-image measurements, including conservative profiler coverage."""
import hashlib
import json
import time
import torch
from torch.nn import functional as F
from .data import load_tensor, image_paths
from .engine import seed_all, write_json


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def forward_profile(adapter, x):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        with torch.no_grad():
            adapter(x)
        synchronize(x.device)
    events = profiler.key_averages()
    unsupported = sorted({e.key for e in events if e.key.startswith("aten::") and not e.flops})
    return {"supported_forward_flops": sum(e.flops for e in events),
            "coverage": "partial_supported_operators", "unaccounted_operators": unsupported,
            "FMA": 2, "profiler": "torch.profiler with_flops", "precision": "FP32",
            "training_step_flops": None, "training_coverage": "NOT_MEASURED"}


def evaluate(args, contract, adapter, manifest):
    from pytorch_msssim import ssim, ms_ssim
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Required LPIPS package and pretrained AlexNet asset missing") from exc
    if not args.resume:
        raise ValueError("Evaluation requires --resume pointing to selected fresh best.pt")
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    progress = checkpoint["sampler_and_progress"]
    minimum = contract["training"]["early_stopping"]["minimum_epochs"]
    latest_path = args.resume.parent/"latest.pt"
    if not latest_path.exists():
        raise ValueError("Selected checkpoint requires completed latest.pt in same run")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if not latest["sampler_and_progress"]["training_complete"] or latest["sampler_and_progress"]["complete_epochs"] < minimum:
        raise ValueError("Incomplete fresh training cannot enter benchmark evaluation")
    identity = checkpoint["identity"]
    if identity != latest["identity"] or identity["model"] != args.model or identity["channel"] != args.channel or identity["source_commit"] != manifest["commit"]:
        raise ValueError("Selected checkpoint model/channel/source identity mismatch")
    if identity["contract_sha256"] != hashlib.sha256(args.contract.read_bytes()).hexdigest():
        raise ValueError("Evaluation contract differs from training")
    if identity["dataset_sha256"]!=args.dataset_sha256:
        raise ValueError("Evaluation training/validation source identity differs from selected checkpoint")
    if progress["complete_epochs"] != checkpoint["early_stop"]["best_epoch"]:
        raise ValueError("Evaluation must use validation-selected best checkpoint")
    adapter.load_state_dict(checkpoint["model"], strict=True)
    adapter.to(args.device).eval()
    if hasattr(adapter,"configure_evaluation"):
        adapter.configure_evaluation(contract)
    perceptual = lpips.LPIPS(net="alex").to(args.device).eval()
    paths = image_paths(args.kodak_root, 24)
    expected_names = {f"kodim{i:02d}" for i in range(1,25)}
    if {p.stem for p in paths} != expected_names:
        raise ValueError("Expected original named Kodak24 images")
    snrs = contract["evaluation"]["snr_db"]
    trials = contract["evaluation"]["independent_channel_trials_per_image"]
    if snrs != [0,5,10,15,20] or trials != 10:
        raise ValueError("Approved full evaluation requires 5 SNR points and 10 independent trials")
    rows_path = args.output / "measured_results.jsonl"
    if rows_path.exists():
        raise ValueError("Evaluation output already exists; use a new output directory")
    profiles = {}
    selected_buffers = {k:v.clone() for k,v in adapter.named_buffers()}
    def restore_selected_buffers():
        for name,value in adapter.named_buffers():
            if name in selected_buffers and value.shape==selected_buffers[name].shape:
                value.copy_(selected_buffers[name])
    with rows_path.open("w", encoding="utf-8") as stream, torch.no_grad():
        for path in paths:
            original = load_tensor(path).unsqueeze(0).to(args.device)
            h,w = original.shape[-2:]
            if (h,w) not in {(512,768),(768,512)}:
                raise ValueError("Kodak image must retain original 512x768 or 768x512 geometry")
            pad_h, pad_w = (-h)%adapter.alignment, (-w)%adapter.alignment
            x = F.pad(original, (0,pad_w,0,pad_h), mode="replicate")
            geometry = f"{x.shape[-2]}x{x.shape[-1]}"
            # Warm actual operators at geometry outside timing/profiling.
            restore_selected_buffers()
            adapter(x)
            restore_selected_buffers()
            if geometry not in profiles:
                profiles[geometry] = forward_profile(adapter, x)
                restore_selected_buffers()
                write_json(args.output/"profiling.json", profiles)
            for snr in snrs:
                adapter.snr_db = snr
                for trial in range(trials):
                    key = f"{identity['seed']}|{path.stem}|{args.channel}|{snr}|{trial}"
                    noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
                    seed_all(noise_seed)
                    # Upstream Q modem mutates annealing steps even under eval.
                    # Restore buffers for every independent realization.
                    restore_selected_buffers()
                    if hasattr(adapter,"transmit"):
                        reconstructed,uses,timings,ledger=adapter.transmit(x,snr,noise_seed)
                        if ledger.get("all_transmitted_streams_counted") is not True or ledger.get("complex_channel_uses")!=uses:
                            raise RuntimeError("Custom adapter communication ledger incomplete or inconsistent")
                    else:
                        synchronize(args.device)
                        start = time.perf_counter()
                        symbols = adapter.encode(x)
                        synchronize(args.device)
                        encoded = time.perf_counter()
                        received = adapter.transport(symbols, snr)
                        synchronize(args.device)
                        transported = time.perf_counter()
                        reconstructed = adapter.decode(received, x.shape)
                        synchronize(args.device)
                        decoded = time.perf_counter()
                        uses=int(adapter.transport.last_complex_uses)
                        timings={"encoder_seconds":encoded-start,"channel_seconds":transported-encoded,
                            "decoder_seconds":decoded-transported,"end_to_end_seconds":decoded-start}
                        ledger={"complex_channel_uses":uses,"serialized_bits":None,
                            "all_transmitted_streams_counted":True,
                            "streams":[{"name":"image_latent","type":"complex_symbols","complex_channel_uses":uses,"serialized_bits":None}]}
                    restore_selected_buffers()
                    rec = reconstructed[:,:,:h,:w].clamp(0,1)
                    if not torch.isfinite(rec).all():
                        raise RuntimeError("Nonfinite reconstruction")
                    mse = float((rec.double()-original.double()).square().mean())
                    row = {"measured":True, "model":args.model,"channel":args.channel,
                        "snr_db":snr,"image":path.stem,"trial":trial,"seed":key,
                        "PSNR":float(-10*torch.log10(torch.tensor(mse))) if mse else None,
                        "SSIM":float(ssim(rec,original,data_range=1.)),
                        "MS_SSIM":float(ms_ssim(rec,original,data_range=1.)),"MSE":mse,
                        "LPIPS":float(perceptual(rec*2-1,original*2-1).mean()),
                        "complex_channel_uses":uses,"CBR":uses/(3*h*w),
                        **timings,
                        "original_height":h,"original_width":w,"padding_bottom":pad_h,"padding_right":pad_w,
                        "native_full_image":True,"serialized_bits":ledger.get("serialized_bits"),"stream_ledger":ledger,
                        "all_transmitted_streams_counted":True,"source_commit":manifest["commit"],
                        "forward_flops_supported":profiles[geometry]["supported_forward_flops"],
                        "flops_coverage":"partial_supported_operators"}
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                    stream.flush()
    expected = 24*len(snrs)*trials
    checkpoint_sha = hashlib.sha256(args.resume.read_bytes()).hexdigest()
    receipt = {"status":"complete","model":args.model,"channel":args.channel,
               "evaluation_complete":True,"measured_rows":expected,
               "eligible_measured":True,"ranking_eligible":False,
               "eligibility_blockers":["Training and forward FLOP coverage is incomplete; no complete-total ranking permitted"],
               "comparison_track":"explicit_matched_channel_extension",
               "author_paper_reproduction":False,"identity":identity,"profiles":profiles,
               "selected_epoch":progress["complete_epochs"],
               "complete_training_epochs":latest["sampler_and_progress"]["complete_epochs"],
               "validation_pass":True,"checkpoint_sha256":checkpoint_sha,
               "early_stopping":latest["early_stop"],
               "training_step_profile":latest["sampler_and_progress"].get("training_step_profile"),
               "stream_ledger":"Full measured stream ledger in every measured_results.jsonl row",
               "source_provenance":manifest["provenance"],
               "reviewed_provenance":False,
               "all_transmitted_streams_counted":True}
    write_json(args.output/"measurement_receipt.json", receipt)
    return receipt
