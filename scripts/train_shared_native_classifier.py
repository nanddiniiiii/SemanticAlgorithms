"""Train the shared-codebook paper's missing STL10 classifier prerequisite.

This is auxiliary classifier training, not a completed semantic codec. Native
architecture, resize, SGD schedule, batch size and 200-epoch target are retained.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--pilot-steps', type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torchvision
    from torchvision.datasets import STL10
    from src.artifacts import write_json
    from src.sources import fetch_source, source_path
    assert torch.cuda.is_available(), 'Native-size classifier training requires actual CUDA hardware'
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = fetch_source('shared_semantic_codebook')
    source = source_path('shared_semantic_codebook')/'googlenet_train.py'
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {'conv_relu', 'inception', 'GoogLeNet', 'data_tf'}]
    assert len(selected) == 4
    namespace = {'torch': torch, 'nn': torch.nn, 'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    random.seed(1024); np.random.seed(1024); torch.manual_seed(1024)
    dataset = STL10(str(args.data_root), split='train', transform=namespace['data_tf'], download=False)
    assert len(dataset) == 5000
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False)
    model = namespace['GoogLeNet'](3, 10).cuda()
    write_json(args.output/'protocol.json', {'source_commit': receipt['commit'],
        'scope': 'native auxiliary classifier prerequisite; not trained semantic codec',
        'dataset': 'STL10 official 5000 labelled training images', 'input_shape': [3, 96, 96],
        'batch_size': 64, 'epochs_target': args.epochs, 'seed': 1024,
        'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'precision': 'float32', 'multi_gpu': False,
        'adaptations': ['modern Torch environment', 'explicit official training split',
            'omit source validation-on-training metric; no test-based checkpoint selection',
            'explicit seed and bounded runtime; optimizer/RNG checkpoint export'],
        'preprocessing': 'native data_tf: PIL bilinear resize96; float32 RGB/255; no augmentation'})
    start = time.monotonic(); steps = images_seen = 0
    epoch = -1; epoch_complete = False; losses = []
    for epoch in range(args.epochs):
        lr = .02 if epoch < 20 else .01 if epoch < 40 else .005 if epoch < 80 else .002 if epoch < 120 else .001 if epoch < 150 else .0005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.train(); total = correct = 0; loss_sum = 0.; epoch_complete = True
        for batch_index, (image, label) in enumerate(loader):
            image = image.cuda(non_blocking=True); label = label.cuda(non_blocking=True)
            optimizer.zero_grad(); logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, label)
            assert torch.isfinite(loss), 'Nonfinite classifier loss'
            loss.backward()
            assert torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
            optimizer.step(); steps += 1; images_seen += len(image)
            loss_sum += float(loss.detach())*len(image); total += len(image)
            correct += int((logits.argmax(1) == label).sum())
            if batch_index == 0:
                print(json.dumps({'stage': 'train', 'epoch': epoch+1, 'step': steps, 'loss': float(loss.detach())}), flush=True)
            if (args.pilot_steps and steps >= args.pilot_steps) or time.monotonic()-start >= args.max_seconds:
                epoch_complete = batch_index+1 == len(loader); break
        record = {'epoch': epoch+1, 'epoch_complete': epoch_complete, 'images': total,
            'training_loss': loss_sum/total, 'training_accuracy': correct/total,
            'steps': steps, 'learning_rate': lr, 'elapsed_seconds': time.monotonic()-start}
        losses.append(record)
        with (args.output/'epochs.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch+1, 'epoch_complete': epoch_complete, 'steps': steps,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'training_images_seen': images_seen, 'purpose': 'STL10 shared-codebook classifier',
            'resume_supported': False}
        temporary = args.output/'classifier.pt.tmp'; torch.save(state, temporary)
        temporary.replace(args.output/'classifier.pt')
        if args.pilot_steps or not epoch_complete or time.monotonic()-start >= args.max_seconds: break
    model.eval()
    loaded = torch.load(args.output/'classifier.pt', map_location='cuda:0', weights_only=True)
    model.load_state_dict(loaded['state_dict'], strict=True)
    with torch.inference_mode(): assert torch.isfinite(model(image)).all()
    completed = epoch+1 == args.epochs and epoch_complete
    status = 'pilot_completed' if args.pilot_steps else 'completed' if completed else 'budget_stopped'
    write_json(args.output/'status.json', {'status': status, 'epochs_entered': epoch+1,
        'last_epoch_complete': epoch_complete, 'optimizer_steps': steps,
        'training_images_seen': images_seen, 'strict_reload': True,
        'elapsed_seconds': time.monotonic()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'semantic_codec_trained': False, 'next': 'construct native semantic codebook; train selected published codec variant'})
    print('SHARED_CLASSIFIER_'+status.upper(), flush=True)


if __name__ == '__main__':
    main()
