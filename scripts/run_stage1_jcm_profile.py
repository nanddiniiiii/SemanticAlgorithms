"""Native JCM pretrained T4 profiling; preserve batch-wide power semantics."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from src.artifacts import sha256, write_json
    from src.codecs import build_codec, upstream_import
    from src.config import Config
    from src.sources import fetch_source, source_path
    from src.metrics import profile_forward
    from src.compute import registered_kernel_flops
    assert torch.cuda.is_available()
    assert sha256(args.checkpoint) == 'b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897'
    args.output.mkdir(parents=True, exist_ok=False)
    fetch_source('jcm')
    torch.manual_seed(42)
    codec = build_codec(Config('jcm',device='cuda:0',snr_db=12,checkpoint=str(args.checkpoint))).eval()
    with upstream_import(source_path('jcm'), ('network','modules')):
        normalize = importlib.import_module('modules').normalize
    data = CIFAR10(args.data_root,train=False,download=False,transform=ToTensor())
    assert len(data) == 10000
    batch = torch.stack([data[i][0] for i in range(32)]).cuda()
    write_json(args.output/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'execution_device':'cuda:0','data':'first32 official CIFAR10 test images; profiling only',
        'scope':'no new training or quality reproduction; native stochastic modulation retained'})
    write_json(args.output/'model.json',codec.metadata)

    class Full(torch.nn.Module):
        def __init__(self):super().__init__();self.core=codec
        def forward(self,x):return self.core(x)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=codec.core.encoder;self.prob=codec.core.prob_convs
        def forward(self,x):
            features=self.encoder(x*2-1).reshape(len(x),-1)
            logits=self.prob(features).reshape(len(x),-1,codec.core.num_category)
            return normalize(codec.core.reparameterize(logits))[1]

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.recon=codec.core.decoder_recon;self.classify=codec.core.decoder_class
        def forward(self,z):return self.recon(z),self.classify(z)

    for size in [1,32]:
        image=batch[:size]
        with torch.inference_mode():
            native=codec.core(image*2-1)
            received=native[2].detach()
            assert native[-1].shape==image.shape
            assert all(torch.isfinite(value).all() for value in native)
        for name,module,value in [('full',Full(),image),('encoder',Encoder(),image),('decoder',Decoder(),received)]:
            print({'stage':'profile','component':name,'batch':size,'input_shape':list(value.shape)},flush=True)
            result=profile_forward(module,value,warmup=10,repeats=100)
            result['registered_kernel_count']=registered_kernel_flops(module,value)
            result['flops_scope']=name+' native16QAM forward; decoder includes reconstruction and classification; encoder includes probability head/modulation/normalization'
            result['memory_note']='Full codec and first32 test tensors remain resident; not isolated component deployment memory'
            result['input_shape']=list(value.shape)
            result['complex_uses_per_image']=128
            write_json(args.output/(name+'_batch'+str(size)+'.json'),result)
    write_json(args.output/'status.json',{'status':'completed','scope':'inference profiles at batch1 and native batch32; no new quality result or training'})
    print('JCM native profiling completed',flush=True)


if __name__ == '__main__':main()
