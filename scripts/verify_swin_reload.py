"""Verify strict retained Swin checkpoint reload after resolution changes."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import gc

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'.runtime_deps')]
import torch
from semantic_lossless.codecs import build_codec
from semantic_lossless.config import Config
from semantic_lossless.artifacts import write_json


def main():
    torch.set_num_threads(2)
    config=Config('swinjscc',rate=96)
    codec=build_codec(config).eval()
    image=torch.rand(1,3,128,128)
    with torch.inference_mode():
        torch.manual_seed(42)
        expected=codec(image)['reconstruction']
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'resolution_checkpoint.pt'
        torch.save({'state_dict':codec.state_dict()},path)
        del codec
        gc.collect()
        restored=build_codec(replace(config,checkpoint=str(path))).eval()
        with torch.inference_mode():
            torch.manual_seed(42)
            actual=restored(image)['reconstruction']
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            larger=restored(torch.rand(1,3,256,256))['reconstruction']
            assert larger.shape==(1,3,256,256) and torch.isfinite(larger).all()
        write_json(ROOT/'Results/implementation_checks/swin_resolution_reload.json',
            {'status':'passed','scope':'CPU synthetic retained-checkpoint reload across regenerated window-mask shapes; not training reproduction',
             'same_rng_reconstruction_max_error':float((actual-expected).abs().max()),
             'input_shapes':[[1,3,128,128],[1,3,256,256]],
             'checkpoint_validation':restored.metadata['adapted_checkpoint_validation']})
    print('Swin resolution-dependent checkpoint reload passed')


if __name__=='__main__':main()
