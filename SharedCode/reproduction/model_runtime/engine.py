"""Complete, interruptible training using fresh experiment identities."""
import copy
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .adapters import reduce_loss
from .data import load_tensor, epoch_order
from .state import EarlyStop, assert_epoch_coverage, atomic_save, capture_rng, restore_rng


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value
    tmp.write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def batch(paths, indices, adapter, epoch, seed, device, validation=False):
    tensors = [load_tensor(paths[i], crop=adapter.crop,
        seed=seed + epoch*1000003 + i, center=validation,
        author_resize=adapter.name == "DeepJSCC") for i in indices]
    return torch.stack(tensors).to(device)


def validate(model, adapter, paths, device, seed):
    rng = capture_rng()
    # The released Q quantizer advances its anneal buffer even in eval mode.
    # Restore every buffer afterwards: validation must not modify training state.
    buffers = {k: v.detach().clone() for k, v in adapter.named_buffers()}
    total = 0.
    try:
        model.eval()
        with torch.no_grad():
            for i in range(len(paths)):
                seed_all(seed+i)
                x = batch(paths, [i], adapter, 0, seed, device, validation=True)
                loss = float(reduce_loss(model(x)))
                if not math.isfinite(loss):
                    raise RuntimeError("Nonfinite validation objective")
                total += loss
    finally:
        for key, value in adapter.named_buffers():
            value.copy_(buffers[key])
        restore_rng(rng)
        model.train()
    return total / len(paths)


def make_optimizers(adapter):
    if hasattr(adapter,"configure_optimizers"):
        return adapter.configure_optimizers()
    optimizer=torch.optim.Adam(adapter.parameters(),lr=1e-4,
        weight_decay=5e-4 if adapter.name=="DeepJSCC" else 0.)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=15)
        if adapter.name=="DeepJSCC" else
        torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda _: .8)
        if adapter.name=="DeepJSCC_Q" else None)
    return [optimizer],[scheduler] if scheduler else []


def applied_step(model,adapter,optimizers,x):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    with torch.profiler.record_function("measured_forward"):
        result=model(x)
        loss=reduce_loss(result)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training objective")
    with torch.profiler.record_function("measured_backward"):
        loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or nonfinite author model gradients")
    clip=getattr(adapter,"gradient_clip_max_norm",None)
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
    with torch.profiler.record_function("measured_optimizer"):
        optimizers[0].step()
    if hasattr(adapter,"auxiliary_loss"):
        if len(optimizers)!=2:
            raise ValueError("Auxiliary author loss requires its separate optimizer")
        optimizers[1].zero_grad(set_to_none=True)
        with torch.profiler.record_function("measured_auxiliary_backward_optimizer"):
            auxiliary=adapter.auxiliary_loss()
            if not torch.isfinite(auxiliary):
                raise RuntimeError("Nonfinite auxiliary loss")
            auxiliary.backward()
            optimizers[1].step()
    return result,loss


def train(args, contract, adapter, manifest, train_paths, val_paths, dataset_hash):
    output = args.output
    device = args.device
    settings = contract["training"]
    seed = settings.get("seed", 20261020)
    expected = settings.get("unique_source_images_per_epoch", 800)
    global_batch = settings.get("global_batch_size", 16)
    if expected != 800 or len(val_paths) != 100 or global_batch != 16:
        raise ValueError("Approved benchmark requires full 800/100 sources and global batch 16")
    if args.device != "cuda" or torch.cuda.device_count() < 2:
        raise RuntimeError("Benchmark training requires two CUDA GPUs; CPU is diagnostic only")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn-only preserves explicit report of nondeterministic native kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = torch.nn.DataParallel(adapter.to(device), device_ids=[0, 1])
    optimizers,schedulers=make_optimizers(adapter)
    optimizer=optimizers[0]
    scheduler=schedulers[0] if schedulers else None
    identity = {"model": args.model, "channel": args.channel, "stage": args.stage,
                "source_commit": manifest["commit"], "dataset_sha256": dataset_hash,
                "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
                "global_batch": global_batch, "seed": seed, "training_snr_db": adapter.snr_db,
                "training_geometry": [adapter.crop, adapter.crop], "precision": "FP32"}
    if hasattr(adapter,"target_cbr"):
        identity.update(target_cbr=adapter.target_cbr,bottleneck_channels=adapter.bottleneck_channels)
    if hasattr(adapter,"protocol_identity"):
        identity["model_protocol"]=adapter.protocol_identity
    early = settings.get("early_stopping", {})
    stopper = EarlyStop(early.get("minimum_epochs", 50), early.get("maximum_epochs", 150),
                        early.get("patience", 20), early.get("relative_min_delta", .001))
    state = {"complete_epochs": 0, "cursor": 0, "order": epoch_order(expected, seed, 0),
             "seen": [], "loss_sum": 0., "history": [], "global_step": 0,
             "elapsed_training_seconds": 0., "training_complete": False}
    seed_all(seed)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != 1 or checkpoint.get("identity") != identity:
            raise ValueError("Resume identity differs; historical checkpoints cannot count as fresh epochs")
        adapter.load_state_dict(checkpoint["model"], strict=True)
        if len(optimizers)!=len(checkpoint["optimizers"]) or len(schedulers)!=len(checkpoint["schedulers"]):
            raise ValueError("Resume requires every author optimizer and scheduler")
        for current,saved in zip(optimizers,checkpoint["optimizers"]):
            current.load_state_dict(saved)
        for current,saved in zip(schedulers,checkpoint["schedulers"]):
            current.load_state_dict(saved)
        stopper = EarlyStop.from_state(checkpoint["early_stop"])
        state = checkpoint["sampler_and_progress"]
        if (not 0<=state["cursor"]<=expected or state["cursor"]%global_batch or
            sorted(state["order"])!=list(range(expected)) or
            state["seen"]!=state["order"][:state["cursor"]] or
            not 0<=state["complete_epochs"]<=stopper.max_epochs):
            raise ValueError("Corrupt resume sampler cursor/order/coverage")
        restore_rng(checkpoint["rng"])
    elif (output / "latest.pt").exists():
        raise ValueError("Existing fresh run found: pass --resume latest.pt or use a new output directory")
    started = time.monotonic()
    old_elapsed = state["elapsed_training_seconds"]
    last_save = started
    interval = min(300., float(settings.get("checkpoint_interval_seconds", 300)))

    def save(name="latest.pt"):
        nonlocal last_save
        state["elapsed_training_seconds"] = old_elapsed + time.monotonic()-started
        payload = {"schema_version": 1, "identity": identity, "model": adapter.state_dict(),
                   "optimizers": [o.state_dict() for o in optimizers],
                   "schedulers": [s.state_dict() for s in schedulers],
                   "rng": capture_rng(), "early_stop": stopper.state_dict(),
                   "sampler_and_progress": copy.deepcopy(state), "fresh_run": True,
                   "upstream_architecture_unchanged": True, "transport_extension": True}
        atomic_save(payload, output/name)
        write_json(output/"progress.json", {**identity, **state, "early_stop": stopper.state_dict(),
                    "status": "TRAINING_COMPLETE" if state["training_complete"] else "RESUMABLE_INCOMPLETE",
                    "training_step_profile": state.get("training_step_profile"),
                    "flops_coverage": "partial_supported_operators" if state.get("training_step_profile") else "NOT_PROFILED"})
        last_save = time.monotonic()

    model.train()
    while not state["training_complete"]:
        epoch = state["complete_epochs"]
        while state["cursor"] < expected:
            indices = state["order"][state["cursor"]:state["cursor"]+global_batch]
            x = batch(train_paths, indices, adapter, epoch, seed, device)
            do_profile = not state.get("training_step_profile")
            activities = [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            context = torch.profiler.profile(activities=activities,with_flops=True) if do_profile else nullcontext()
            with context as profiler:
                result,loss=applied_step(model,adapter,optimizers,x)
            if do_profile:
                events = profiler.key_averages()
                state["training_step_profile"] = {
                    "actual_forward_backward_optimizer_step":True,"batch_size":len(indices),
                    "supported_training_step_flops":sum(e.flops for e in events),
                    "coverage":"partial_supported_operators","FMA":2,
                    "unaccounted_operators":sorted({e.key for e in events if e.key.startswith("aten::") and not e.flops}),
                    "profiler":"torch.profiler with_flops, includes actual optimizer step; DP worker threads may not be captured",
                    "optimizer_updates_per_step":len(optimizers),"precision":"FP32"}
                write_json(output/"training_profile.json",state["training_step_profile"])
            state["seen"].extend(indices)
            state["cursor"] += len(indices)
            state["global_step"] += 1
            state["loss_sum"] += float(loss.detach()) * len(indices)
            if getattr(adapter,"scheduler_step_per_batch",False):
                for schedule in schedulers:
                    schedule.step()
            if time.monotonic()-last_save >= interval:
                save()
            if args.chunk_seconds and time.monotonic()-started >= args.chunk_seconds:
                save()
                return {"status": "RESUMABLE_INCOMPLETE", "complete_epochs": state["complete_epochs"]}
        assert_epoch_coverage(state["seen"], expected)
        validation = validate(model, adapter, val_paths, device, seed+90000000)
        complete = epoch+1
        improved, stop = stopper.update(complete, validation)
        if scheduler and args.model == "DeepJSCC":
            scheduler.step(validation)
        elif scheduler and not getattr(adapter,"scheduler_step_per_batch",False) and stopper.bad_epochs and stopper.bad_epochs % 10 == 0:
            scheduler.step()
        state["history"].append({"epoch": complete, "sources": expected,
            "train_objective": state["loss_sum"]/expected, "validation_objective": validation,
            "learning_rate": optimizer.param_groups[0]["lr"]})
        state.update(complete_epochs=complete, cursor=0, seen=[], loss_sum=0.,
                     order=epoch_order(expected, seed, complete), training_complete=stop)
        save()
        if improved:
            save("best.pt")
        print(json.dumps(state["history"][-1]), flush=True)
    return {"status": "TRAINING_COMPLETE", "training_complete":True,"complete_epochs": state["complete_epochs"],
            "selected_epoch": stopper.best_epoch, "selected_validation_objective": stopper.best}
