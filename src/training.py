"""Native-objective training. Every run writes new checkpoints, never inputs."""
import time
import random
from .artifacts import write_json


def train(config, codec, run, events, pilot=False):
    import torch
    import numpy as np
    from .data import loader
    if not codec.trainable:
        raise NotImplementedError(f'{config.algorithm} has no verified training adapter')
    train_loader = loader(config, 'train')
    validation = loader(config, 'val')
    optimizer = codec.optimizer()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs + 1, eta_min=1e-6)
    use_amp = config.amp and config.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    model = codec
    if config.multi_gpu:
        if torch.cuda.device_count() < 2 or config.device not in ('cuda', 'cuda:0'):
            raise RuntimeError('multi_gpu requires at least two CUDA devices and primary cuda:0')
        model = torch.nn.DataParallel(codec)
    start = time.monotonic()
    step = 0
    best = float('inf')
    target_steps = 2 if pilot else config.max_steps
    epochs = 1 if pilot else config.epochs
    stop = False
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images_seen = 0
        epoch_complete = True
        for index, batch in enumerate(train_loader):
            batch = {k: v.to(config.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            remaining = len(train_loader) - (index // config.accumulation_steps) * config.accumulation_steps
            divisor = min(config.accumulation_steps, remaining)
            with torch.autocast(device_type='cuda' if config.device.startswith('cuda') else 'cpu', enabled=use_amp):
                output = model(batch['image'])
                loss = codec.loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss / divisor).backward()
            images_seen += len(batch['image'])
            if (index + 1) % config.accumulation_steps == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                for parameter in codec.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Non-finite gradient')
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                events.emit('pilot' if pilot else 'train', metrics={'loss': float(loss.detach()), 'images_seen_epoch': images_seen}, epoch=epoch + 1, step=step)
                if (target_steps and step >= target_steps) or (config.max_seconds and time.monotonic() - start >= config.max_seconds):
                    stop = True
                    epoch_complete = index + 1 == len(train_loader)
                    break
        model.eval()
        validation_loss, count = 0., 0
        with torch.inference_mode():
            for index, batch in enumerate(validation):
                batch = {k: v.to(config.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss = codec.loss(model(batch['image']), batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite validation loss')
                validation_loss += float(loss) * len(batch['image'])
                count += len(batch['image'])
                if pilot or (config.max_seconds and time.monotonic() - start >= config.max_seconds):
                    break
        validation_loss /= max(count, 1)
        validation_complete = count == len(validation.dataset) if hasattr(validation, 'dataset') else not pilot
        scheduler.step(epoch + 1)
        state = {'state_dict': codec.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'epoch_complete': epoch_complete, 'step': step,
                 'config': config.as_dict(), 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                 'numpy_rng': {'generator':np.random.get_state()[0], 'keys':np.random.get_state()[1].tolist(), 'position':np.random.get_state()[2], 'has_gauss':np.random.get_state()[3], 'cached_gaussian':np.random.get_state()[4]}, 'validation_loss': validation_loss, 'validation_complete':validation_complete,
                 'resume_supported': False, 'reason': 'Restart from a new run; exact mid-epoch resume is not implemented'}
        temporary = run / 'last.pt.tmp'
        torch.save(state, temporary)
        temporary.replace(run / 'last.pt')
        # Verify a strict round trip before calling the pilot successful.
        if pilot:
            loaded = torch.load(run / 'last.pt', map_location=config.device, weights_only=True)
            codec.load_state_dict(loaded['state_dict'], strict=True)
        if validation_loss < best and (validation_complete or pilot):
            best = validation_loss
            temporary = run / 'best.pt.tmp'
            torch.save({'state_dict': codec.state_dict(), 'config': config.as_dict(), 'epoch': epoch + 1}, temporary)
            temporary.replace(run / 'best.pt')
        events.emit('validation', 'completed', metrics={'native_objective': validation_loss, 'images_evaluated': count}, epoch=epoch + 1, epoch_complete=epoch_complete, validation_complete=validation_complete)
        if stop:
            break
    status = 'pilot_completed' if pilot else ('budget_stopped' if stop and (not epoch_complete or epoch + 1 < config.epochs) else 'completed')
    write_json(run / 'status.json', {'status': status, 'optimizer_steps': step, 'epochs_entered': epoch + 1, 'last_epoch_complete': epoch_complete})
    events.emit('pilot' if pilot else 'train', status, metrics={'optimizer_steps': step, 'elapsed_seconds': time.monotonic() - start})
    return status
