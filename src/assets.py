"""Download only explicitly identified author checkpoints; verify known digests."""
from pathlib import Path
import stat
import urllib.request
import zipfile

from .dataset import digest

ASSETS = {
    'adjscc': {'filename':'ADJSCC_C=2.pth.tar',
        'folder':'https://drive.google.com/drive/folders/1N0EzzxCv1wh6JeFr0g8vkmB0Qj23ozZJ',
        'sha256':'c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54'},
    'swinjscc': {'filename':'SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C32.model',
        'folder':'https://drive.google.com/drive/folders/1_EouRY4yYvMCtamX2ReBzEd5YBQbyesc',
        'sha256':'864d52a64281b3fcd4db34f7696b225b96c998b040b81d96cc556ecb837a17ff'},
    'jcm': {'filename':'CIFAR_SNR12_Trans128_16qam.pth.tar',
        'url':'https://raw.githubusercontent.com/SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication/9a53ef0a3df2f41ecae6cd531968f70a13d53088/checkpoints/16qam/CIFAR_SNR12.000_Trans128_16qam.pth.tar',
        'sha256':'b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897'}
}

# Digests of the ten files extracted from the author-linked archive, recorded in
# Docs/PRETRAINED_DOWNLOADS.md. A matching directory name alone proves nothing.
SQGAN_DIGESTS={
    'decoder.ckpt':'a6af82ae24c772790757f93861aa7ac835a684e8251eaeb597a5ecdee39bfe8f',
    'decoder_semantic.ckpt':'28e1f6f400c3f09f01c77c11a1190b800e7886323f30b876e47bd8ad642b462e',
    'demasker.ckpt':'87a83124c20d839f6fef65f724d86dc9445ab638994834d58bbf7dd3a5408b65',
    'demasker_semantic.ckpt':'d6d6dad1331ff8a832f20ad5e0256d6d341cb82d39fcd77a09c7bca54e6dcf1d',
    'encoder.ckpt':'6afacd86a809fb8f2e2f898934a4a5e8797ac034a50ad90234d12ef36dba0268',
    'encoder_semantic.ckpt':'e40534ce8c0335d01e4a201fafc47121dc5102f2a217e525cb9973facd61572f',
    'masker.ckpt':'da3187ab8165c0151f3e7e1efa08f9342c10ec89cfbd7ab03cad734f822ba678',
    'masker_semantic.ckpt':'0932245c0a1a3e82fb44d99c87017e5983fddd6e558f8966624be7cebe5fda99',
    'quantize.ckpt':'10bf41064c07bca2d46a3bdd06799f28e48212813553f51028f5be38847bee95',
    'quantize_semantic.ckpt':'46c4968df648b214c86ca25e648612c59107ab395c01e8819531db45d2dbc20c'}


def verified_sqgan_components(path):
    path=Path(path)
    return path.is_dir() and all((path/name).is_file() and digest(path/name)==expected
                                 for name,expected in SQGAN_DIGESTS.items())


def obtain(name, cache, search_roots=()):
    if name=='sqgan':return obtain_sqgan(cache,search_roots)
    if name not in ASSETS:
        raise ValueError('No automatically retrievable checkpoint configured for '+name)
    spec=ASSETS[name];cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    target=cache/spec['filename']
    for root in [cache,*map(Path,search_roots)]:
        if not root.exists():continue
        for path in root.rglob(spec['filename']):
            if digest(path)==spec['sha256']:
                print('Verified checkpoint: '+str(path),flush=True);return path
    temporary=target.with_suffix(target.suffix+'.partial')
    if spec.get('url'):
        with urllib.request.urlopen(spec['url'],timeout=90) as incoming, temporary.open('wb') as stream:
            while block:=incoming.read(1024*1024):stream.write(block)
    else:
        import gdown
        entries=gdown.download_folder(url=spec['folder'],skip_download=True,quiet=False)
        matches=[entry for entry in entries or [] if Path(entry.path).name==spec['filename']]
        if len(matches)!=1:
            raise FileNotFoundError('Author folder did not expose exactly one '+spec['filename']+'; mount verified weights instead')
        if not gdown.download(id=matches[0].id,output=str(temporary),quiet=False):
            raise RuntimeError('Author checkpoint download failed')
    if digest(temporary)!=spec['sha256']:
        raise ValueError('Downloaded checkpoint digest mismatch; refusing to load '+str(temporary))
    temporary.replace(target)
    return target


def obtain_sqgan(cache,search_roots=()):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    for root in [cache,*map(Path,search_roots)]:
        if root.exists():
            for path in root.rglob('Final_ckpt_parts'):
                if verified_sqgan_components(path):return path
    import gdown
    archive=cache/'Final_ckpt_parts.zip'
    expected='7bc059f3334993f9acd9f215132391c77ffbd1484d4750f93d58dec00152facf'
    if not archive.is_file() or digest(archive)!=expected:
        temporary=archive.with_suffix('.partial')
        if not gdown.download(id='10N0pxmfLbm-D2lYTOX-Y0014YicF2u-q',output=str(temporary),quiet=False):
            raise RuntimeError('SQ-GAN author archive download failed')
        if digest(temporary)!=expected:raise ValueError('SQ-GAN archive digest mismatch')
        temporary.replace(archive)
    destination=(cache/'sqgan_extracted').resolve();destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as contents:
        if sum(item.file_size for item in contents.infolist())>1024**3:
            raise ValueError('SQ-GAN archive unexpectedly exceeds its known extraction size')
        for item in contents.infolist():
            target=(destination/item.filename).resolve();target.relative_to(destination)
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Unexpected symlink in SQ-GAN archive')
            if item.is_dir():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with contents.open(item) as incoming,target.open('wb') as stream:
                while block:=incoming.read(1024*1024):stream.write(block)
    candidates=list(destination.rglob('Final_ckpt_parts'))
    if len(candidates)!=1 or not verified_sqgan_components(candidates[0]):
        raise ValueError('SQ-GAN archive does not contain the expected ten-component set')
    return candidates[0]
