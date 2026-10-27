"""Inference profiling that permits ISEC's required codeword gradients."""
import math
import time


def profile(model,image,predict,warmup=3,repeats=10):
    import torch
    def sync():
        if image.is_cuda:torch.cuda.synchronize(image.device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):predict(model,image)
        sync()
        if image.is_cuda:torch.cuda.reset_peak_memory_stats(image.device)
        baseline=torch.cuda.memory_allocated(image.device) if image.is_cuda else None
        timings=[]
        for _ in range(repeats):
            sync();started=time.perf_counter();predict(model,image);sync()
            timings.append(time.perf_counter()-started)
        peak=torch.cuda.max_memory_allocated(image.device) if image.is_cuda else None
        timings.sort()
        result={'parameters':sum(p.numel() for p in model.parameters()),'input_shape':list(image.shape),
            'precision':str(image.dtype),'device':str(image.device),'warmup':warmup,'repeats':repeats,
            'latency_p50_seconds':timings[len(timings)//2],
            'latency_p95_seconds':timings[min(repeats-1,math.ceil(.95*repeats)-1)] if repeats>=10 else None,
            'peak_cuda_memory_bytes':peak,'incremental_cuda_memory_bytes':peak-baseline if peak is not None else None,
            'includes_iterative_latent_gradients':bool(getattr(model,'needs_grad_evaluation',False)),
            'training_flops_measured':False}
        try:
            from torch.utils.flop_counter import FlopCounterMode
            counter=FlopCounterMode(display=False)
            with counter:predict(model,image);sync()
            result['registered_kernel_flops']=counter.get_total_flops()
            result['flops_coverage']='PyTorch registered kernels; not a complete arithmetic count; 2 FLOPs/MAC'
        except Exception as exc:
            result['registered_kernel_flops']=None;result['flops_error']=str(exc)
        # Separate instrumented pass: synchronized hooks add overhead, so their
        # durations must not be subtracted from the uninstrumented total above.
        candidates={'encoder':'encoder','decoder':'decoder','core.encoder':'encoder','core.decoder':'decoder',
            'core.jscc_encoder':'encoder','core.jscc_decoder':'decoder','core.ga':'encoder','core.gs':'decoder'}
        clocks={};components={};hooks=[]
        def before(name):
            def callback(module,args):sync();clocks[name]=time.perf_counter()
            return callback
        def after(name):
            def callback(module,args,out):
                sync();components.setdefault(name,[]).append(time.perf_counter()-clocks[name])
            return callback
        for name,module in model.named_modules():
            if name in candidates:
                hooks.extend([module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))])
        try:predict(model,image);sync()
        finally:
            for hook in hooks:hook.remove()
        result['component_instrumented_seconds']={name:{'total':sum(values),'calls':len(values),'role':candidates[name]}
                                                  for name,values in components.items()}
        result['component_scope']='One synchronized instrumented pass; missing parent hooks mean not measured (e.g. JCM decoder sublayers called directly)'
    return result
