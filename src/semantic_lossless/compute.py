"""Registered-kernel FLOPs with operation coverage, including transposed convs.

This complements torch.profiler's limited arithmetic estimate. It is not a
claim to count bitstream manipulation, all elementwise work or CPU instructions.
"""


def registered_kernel_flops(model, image):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import FlopCounterMode

    class Observe(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = set()

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            self.operations.add(str(function).rsplit('.', 1)[0])
            return function(*args, **(kwargs or {}))

    observer = Observe()
    counter = FlopCounterMode(display=False)
    model.eval()
    with torch.inference_mode(), observer, counter:
        model(image)
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
    all_counts = counter.get_flop_counts()
    global_counts = {str(operator): int(value) for operator, value in all_counts.get('Global', {}).items()}
    counted = set(global_counts)
    return {
        'registered_kernel_flops': int(counter.get_total_flops()),
        'operator_flops': global_counts,
        'counted_operator_types': sorted(counted),
        'observed_uncounted_operator_types': sorted(observer.operations-counted),
        'module_flops': {name: sum(int(v) for v in values.values()) for name, values in all_counts.items()},
        'coverage': 'PyTorch registered kernels only; convolution including transposed, matrix and supported attention kernels',
        'convention': 'two FLOPs per multiply-accumulate; unregistered elementwise, coding and integer operations not estimated',
        'complete_arithmetic_count': False,
        'input_shape': list(image.shape), 'torch': torch.__version__,
    }
