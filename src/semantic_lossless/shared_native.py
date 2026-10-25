"""Released SAC-without-proposed-loss components, distinct from the common adapter."""
import ast
from datetime import datetime
import importlib
from types import SimpleNamespace

from .sources import fetch_source, source_path
from .codecs import upstream_import


def native_construction_definitions():
    import numpy as np
    import torch
    path=source_path('shared_semantic_codebook')/'SC_construction.py'
    tree=ast.parse(path.read_text())
    names={'conv_relu','inception','GoogLeNet','data_tf','downsampling'}
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert len(nodes)==5
    namespace={'torch':torch,'nn':torch.nn,'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def build_native_sac(device='cuda:0'):
    import torch
    receipt=fetch_source('shared_semantic_codebook')
    directory=source_path('shared_semantic_codebook')
    args=SimpleNamespace(trainset='STL10',testset='STL10',C=4,model='WITT_W/O',
        channel_type='awgn',multiple_snr='10',distortion_metric='MSE',SCsize=10,seed=1024)
    path=directory/'trainSCFading.py'
    tree=ast.parse(path.read_text())
    node=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='config')
    namespace={'torch':torch,'nn':torch.nn,'datetime':datetime,'args':args}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    config=namespace['config']
    config.device=torch.device(device)
    with upstream_import(directory,('net','loss')):
        model=importlib.import_module('net.networkCBFading').WITT(args,config)
    metadata={'source_commit':receipt['commit'],'branch':'released SAC without proposed loss',
        'model':'WITT_W/O','C':4,'input_shape':[3,256,256],'native_batch_size':12,
        'main_complex_uses_per_image':512,'main_cbr_resized_rgb':512/(3*256*256),
        'prototype_entries':10,'prototype_index_bits_fixed_width':4,
        'side_channel':'prototype selection assumed error free; power scale restored inside simulator',
        'channel':'unmodified native batch-wide normalization and complex AWGN10dB',
        'adaptations':['device remapped from source cuda:3 to requested CUDA device',
                       'explicit fixed SNR10 rather than equivalent singleton-list choice',
                       'CLI codebook size10 accepted as documented by author README'],
        'training_reproduction':False,'stopping_rule':'bounded implementation prefix; no paper convergence claim'}
    return model.to(device),metadata
