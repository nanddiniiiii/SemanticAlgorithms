import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires the model Torch environment')
class KernelCounts(unittest.TestCase):
    def test_transposed_convolution_count_includes_all_input_products(self):
        from src.compute import registered_kernel_flops
        model = torch.nn.ConvTranspose2d(3, 2, kernel_size=3, stride=2, padding=0, bias=False)
        result = registered_kernel_flops(model, torch.ones(1, 3, 4, 4))
        # No border cropping:16 input locations *3 inputs *2 outputs *9 taps *2 FLOPs.
        self.assertEqual(result['registered_kernel_flops'], 16*3*2*9*2)
        self.assertFalse(result['complete_arithmetic_count'])
        self.assertTrue(any('convolution' in key for key in result['counted_operator_types']))

    def test_linear_count_and_uncounted_relu_are_explicit(self):
        from src.compute import registered_kernel_flops
        model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False), torch.nn.ReLU())
        result = registered_kernel_flops(model, torch.ones(2, 4))
        self.assertEqual(result['registered_kernel_flops'], 2*4*3*2)
        self.assertTrue(any('relu' in key for key in result['observed_uncounted_operator_types']))
