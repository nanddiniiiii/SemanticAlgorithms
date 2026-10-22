import unittest
try:
    import torch
except ImportError:
    torch = None
from semantic_lossless.metrics import profile_forward, quality


@unittest.skipIf(torch is None, 'Run these checks in the model Torch environment')
class MetricContracts(unittest.TestCase):
    def test_supported_conv_flops_matches_independent_operation_count(self):
        model=torch.nn.Conv2d(3,2,kernel_size=1,bias=False)
        result=profile_forward(model,torch.zeros(1,3,4,4),warmup=0,repeats=1)
        # 16 output locations * 2 output channels * 3 products, two FLOPs/MAC.
        self.assertEqual(result['forward_flops'],16*2*3*2)
        self.assertEqual(result['parameters'],6)
        self.assertIsNone(result['latency_p95_seconds'])

    def test_expensive_profile_does_not_hide_extra_executions(self):
        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__();self.calls=0
            def forward(self,value):
                self.calls+=1
                return value
        model=Counter()
        result=profile_forward(model,torch.zeros(1,3,4,4),warmup=0,repeats=1,count_flops=False)
        self.assertEqual(model.calls,1)
        self.assertIsNone(result['forward_flops'])
        self.assertEqual(result['flops_coverage'],'not_requested')

    def test_uint8_exactness_does_not_claim_float_exactness(self):
        original=torch.full((1,3,16,16),.5)
        result=quality(original,original+.0001)
        self.assertEqual(result['exact_uint8_pixels'],1)
        self.assertEqual(result['exact_float_reconstruction'],0)
        self.assertGreater(result['mse'],0)


if __name__=='__main__':unittest.main()
