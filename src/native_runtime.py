"""Audited compatibility copies; immutable author downloads are never edited."""
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import shutil

_device=ContextVar('semantic_native_device',default='cpu')
def current_device():return _device.get()

def legacy_fft(value, signal_ndim=1, normalized=False):
    import torch
    z=torch.view_as_complex(value.contiguous())
    return torch.view_as_real(torch.fft.fftn(z,dim=tuple(range(-signal_ndim,0)),norm='ortho' if normalized else 'backward'))

def legacy_ifft(value, signal_ndim=1, normalized=False):
    import torch
    z=torch.view_as_complex(value.contiguous())
    return torch.view_as_real(torch.fft.ifftn(z,dim=tuple(range(-signal_ndim,0)),norm='ortho' if normalized else 'backward'))


def runtime_source(name):
    from .sources import fetch_source,source_path
    from .paths import project_root
    from .runtime import write_json
    receipt=fetch_source(name)
    source=source_path(name)
    target=project_root()/'.runtime_sources'/name/(receipt['commit']+'-compat-v3')
    if target.exists():
        ledger=json.loads((target/'compatibility.json').read_text())
        expected={item['path']:item['sha256'] for item in receipt['files']}
        for item in ledger['changes']:expected[item['path']]=item['after']
        for relative,value in expected.items():
            data=(target/relative).read_bytes()
            if hashlib.sha256(data).hexdigest()!=value:
                raise ValueError('Runtime source modified: '+relative)
        return target
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.with_name(target.name+'.building')
    if staging.exists():raise RuntimeError('Interrupted runtime source preparation: '+str(staging))
    shutil.copytree(source,staging)
    changes=[]
    for path in staging.rglob('*.py'):
        original=path.read_text(encoding='utf-8-sig');text=original
        if name=='ofdm_guided_jscc':
            text=text.replace('torch.fft(', 'legacy_fft(').replace('torch.ifft(', 'legacy_ifft(')
            if text!=original:text='from src.native_runtime import legacy_fft, legacy_ifft\n'+text
        if name in {'deepjscc_mimo','ssc'} and '.cuda()' in text:
            text='from src.native_runtime import current_device\n'+text.replace('.cuda()', '.to(current_device())')
        # BasicSR's old torchvision path was renamed; this is an import-only change.
        text=text.replace('torchvision.transforms.functional_tensor','torchvision.transforms._functional_tensor')
        if text!=original:
            path.write_text(text,encoding='utf-8')
            changes.append({'path':path.relative_to(staging).as_posix(),
                'before':hashlib.sha256(original.encode()).hexdigest(),'after':hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(staging/'compatibility.json',{'source_commit':receipt['commit'],'changes':changes,
        'scope':'FFT pair representation and explicit replica device/import compatibility; architecture unchanged'})
    staging.rename(target)
    return target
