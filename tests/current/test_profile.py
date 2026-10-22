import unittest
from pathlib import Path
import tempfile
try:
    import torch
except ImportError:
    torch = None
from semantic_lossless.metrics import profile_forward, quality


@unittest.skipIf(torch is None, 'Run these checks in the model Torch environment')
class MetricContracts(unittest.TestCase):
    def test_cityscapes_validation_matches_pinned_author_loader(self):
        from semantic_lossless.sources import source_path
        directory=source_path('sqgan')
        if not (directory/'data/cityscapes_base.py').is_file():
            self.skipTest('Fetch pinned SQ-GAN source for its independent loader check')
        import importlib
        import numpy as np
        from PIL import Image
        from semantic_lossless.codecs import upstream_import
        from semantic_lossless.config import Config
        from semantic_lossless.data import Cityscapes
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            images=root/'leftImg8bit/val/city';labels=root/'gtFine/val/city'
            images.mkdir(parents=True);labels.mkdir(parents=True)
            image=images/'city_000000_000000_leftImg8bit.png'
            label=labels/'city_000000_000000_gtFine_labelIds.png'
            # Noninteger downsampling and narrow boundaries distinguish PIL
            # nearest from the author's tensor nearest coordinate convention.
            classes=np.broadcast_to((np.arange(1040)//3)%4,(520,1040))
            Image.fromarray(np.random.default_rng(1).integers(0,256,(520,1040,3),dtype=np.uint8)).save(image)
            Image.fromarray(np.asarray([7,8,26,255],dtype=np.uint8)[classes]).save(label)
            reference_label=root/'reference_trainIds.png'
            Image.fromarray(np.asarray([0,1,13,255],dtype=np.uint8)[classes]).save(reference_label)
            actual=Cityscapes(Config('sqgan',dataset='cityscapes',data_root=str(root),
                model_options={'label_format':'label_ids'}),'test')[0]
            with upstream_import(directory,('data',)):
                native=importlib.import_module('data.cityscapes_base').ImagePaths(
                    split='val',is_val=True,paths=[str(image)],
                    elements_input={'file_path_':[str(image)],'ssm_path':[str(reference_label)]},
                    size=256,ratio_dim=2,relevant_obj=[0]*19)[0]
            mean=torch.tensor([.485,.456,.406])[:,None,None]
            std=torch.tensor([.229,.224,.225])[:,None,None]
            torch.testing.assert_close((actual['image']-mean)/std,native['image'],rtol=0,atol=1e-6)
            torch.testing.assert_close(actual['semantic'],native['semantic'],rtol=0,atol=0)

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
