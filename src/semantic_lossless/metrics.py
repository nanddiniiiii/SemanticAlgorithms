"""Quality on RGB [0,1]. Infinity is represented as null plus an exact flag."""
import math


def quality(original, reconstruction, use_lpips=False):
    import torch
    if original.shape != reconstruction.shape:
        raise ValueError("Image shapes differ")
    if not torch.isfinite(original).all() or not torch.isfinite(reconstruction).all():
        raise ValueError("Reconstruction contains non-finite values")
    a = original.float()
    b = reconstruction.float().clamp(0, 1)
    mse = float((a - b).square().mean())
    result = {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse > 0 else None,
              "max_absolute_error": float((a - b).abs().max()),
              "exact_float_reconstruction": int(torch.equal(a, b)),
              "exact_uint8_pixels": int(torch.equal((a*255).round().to(torch.uint8), (b*255).round().to(torch.uint8)))}
    original_samples = (a*255).round().to(torch.int16)
    reconstructed_samples = (b*255).round().to(torch.int16)
    equal_samples = original_samples == reconstructed_samples
    result['exact_uint8_sample_fraction'] = float(equal_samples.float().mean())
    result['exact_uint8_spatial_pixel_fraction'] = float(equal_samples.all(dim=1).float().mean())
    result['max_absolute_uint8_error'] = int((original_samples-reconstructed_samples).abs().max())
    try:
        from pytorch_msssim import ssim
        result["ssim"] = float(ssim(a, b, data_range=1, size_average=True))
    except ImportError:
        result["ssim"] = None
    if use_lpips:
        global _LPIPS
        if "_LPIPS" not in globals():
            import lpips
            _LPIPS = lpips.LPIPS(net="alex").to(a.device).eval()
        with torch.no_grad():
            result["lpips"] = float(_LPIPS(a * 2 - 1, b * 2 - 1).mean())
    return result


def profile_forward(codec, image, warmup=3, repeats=10, count_flops=True):
    import time
    import torch
    if warmup < 0 or repeats < 1:
        raise ValueError('Profiling needs nonnegative warmup and at least one repeat')
    def synchronize():
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
    codec.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            codec(image)
        synchronize()
        if image.is_cuda:
            torch.cuda.reset_peak_memory_stats(image.device)
        baseline_allocated = torch.cuda.memory_allocated(image.device) if image.is_cuda else None
        baseline_reserved = torch.cuda.memory_reserved(image.device) if image.is_cuda else None
        elapsed = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            codec(image)
            synchronize()
            elapsed.append(time.perf_counter() - start)
        peak_allocated = torch.cuda.max_memory_allocated(image.device) if image.is_cuda else None
        peak_reserved = torch.cuda.max_memory_reserved(image.device) if image.is_cuda else None
        flops, counted, coverage, error = None, [], 'not_requested', None
        if count_flops:
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profiler:
                    codec(image)
                    synchronize()
                operations = profiler.key_averages()
                flops = sum(event.flops or 0 for event in operations)
                counted = [event.key for event in operations if event.flops]
                coverage = "partial_supported_operators"
            except Exception as exc:
                flops, counted, coverage, error = None, [], "unavailable", str(exc)
    elapsed.sort()
    return {"latency_p50_seconds": elapsed[len(elapsed)//2],
            "latency_p95_seconds": elapsed[min(len(elapsed)-1, math.ceil(0.95*len(elapsed))-1)] if repeats >= 10 else None,
            "latency_note": 'single observed execution; not a latency distribution' if repeats == 1 else 'empirical repeated executions; p95 omitted below ten observations',
            "parameters": sum(p.numel() for p in codec.parameters()), "forward_flops": flops,
            "flops_scope": "complete codec forward including native channel/classifier where present",
            "flops_coverage": coverage, "counted_operator_types": counted,
            "profiler_error": error, "batch_size": len(image), "precision": str(image.dtype),
            "input_shape": list(image.shape),
            "peak_cuda_memory_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "baseline_cuda_allocated_bytes": baseline_allocated,
            "baseline_cuda_reserved_bytes": baseline_reserved,
            "incremental_peak_cuda_allocated_bytes": peak_allocated-baseline_allocated if image.is_cuda else None,
            "memory_scope": "repeated-forward process residency plus temporary allocations; not isolated component memory; subsequent FLOPs tracing excluded",
            "warmup": warmup, "repeats": repeats, "training_flops_measured": False}
