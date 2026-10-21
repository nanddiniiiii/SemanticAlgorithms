"""Isolated throughput probe. Never install in a production training session."""
import functools
import json
import os
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class Probe(Callback):
    def __init__(self):
        self.times = []
        self.waits = []
        self.previous_end = None
        self.start = None
        self.resume = None
        self.nonfinite = False

    def on_train_start(self, trainer, module):
        self.resume = {'epoch_index': int(trainer.current_epoch), 'global_step': int(trainer.global_step),
                       'optimizers': len(trainer.optimizers), 'schedulers': len(trainer.lr_scheduler_configs),
                       'lr': [[g['lr'] for g in o.param_groups] for o in trainer.optimizers]}
        expected = json.loads(os.environ['PROBE_EXPECTED'])
        assert self.resume['global_step'] == expected['global_step']
        assert self.resume['optimizers'] == self.resume['schedulers'] == 2
        assert self.resume['lr'] == expected['lr']
        assert len(trainer.train_dataloader.dataset) == 800
        assert trainer.num_training_batches == 400
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        if self.previous_end is not None:
            self.waits.append(self.start - self.previous_end)

    def on_before_backward(self, trainer, module, loss):
        assert torch.isfinite(loss).all().item(), 'Non-finite probe loss'

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        end = time.perf_counter()
        self.times.append(end-self.start)
        self.previous_end = end
        if len(self.times) >= 40:
            trainer.should_stop = True

    def on_train_end(self, trainer, module):
        n = 10  # exclude cold-start warmup
        compute = self.times[n:]
        waits = self.waits[n-1:]
        report = {'variant': os.environ['PROBE_VARIANT'], 'rank': trainer.global_rank,
                  'resume': self.resume, 'samples_per_epoch': 800, 'batches_per_full_epoch': 400,
                  'batch_per_gpu': 1, 'precision': str(trainer.precision),
                  'batches_observed': len(self.times), 'warmup_batches': n,
                  'mean_compute_seconds': sum(compute)/len(compute),
                  'mean_input_and_transfer_seconds': sum(waits)/len(waits),
                  'mean_batch_seconds': (sum(compute)+sum(waits))/len(compute),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'all_model_parameters_finite': all(torch.isfinite(p).all().item() for p in module.parameters()),
                  'benchmark_updates_count_as_training': False}
        assert report['all_model_parameters_finite']
        folder=Path(os.environ['PROBE_OUTPUT']); folder.mkdir(parents=True,exist_ok=True)
        (folder/f'rank_{trainer.global_rank}.json').write_text(json.dumps(report,indent=2))
        print('SPEED PROBE RESULT', json.dumps(report), flush=True)


def install():
    from torch.utils.data import DataLoader
    original_loader = DataLoader.__init__
    @functools.wraps(original_loader)
    def loader(self, *args, **kwargs):
        kwargs['pin_memory'] = os.environ.get('PROBE_PIN') == '1'
        if kwargs.get('num_workers', 0) > 0:
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = 2
        return original_loader(self, *args, **kwargs)
    DataLoader.__init__ = loader
    cache_map = os.environ.get('PROBE_CACHE_MAP')
    if cache_map:
        from adaptations.sqgan_div2k_channel.dataset import DIV2KWithPseudoLabels
        original_init = DIV2KWithPseudoLabels.__init__
        mapping=json.loads(Path(cache_map).read_text())
        @functools.wraps(original_init)
        def init(self,*args,**kwargs):
            original_init(self,*args,**kwargs)
            self.pairs=[tuple(Path(v) for v in mapping[str(pair[0])]) for pair in self.pairs]
        DIV2KWithPseudoLabels.__init__=init
    original_fit=pl.Trainer.fit
    def fit(self,*args,**kwargs):
        # Probe checkpoints and logger overhead are excluded consistently.
        self.callbacks=[c for c in self.callbacks if type(c).__name__ not in {'ModelCheckpoint','CaptionImageLogger'}]
        self.callbacks.append(Probe())
        return original_fit(self,*args,**kwargs)
    pl.Trainer.fit=fit
