"""Notebook orchestration for the pinned author Lightning 1.9 training script."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SessionControl(Callback):
    def __init__(self):
        super().__init__()
        self.deadline = float(os.environ.get("SQGAN_SESSION_DEADLINE_UNIX", "inf"))
        self.epoch_limit = int(os.environ.get("SQGAN_STOP_AFTER_STAGE_EPOCHS", "150"))
        self.epoch_started = None

    def on_train_start(self, trainer, pl_module):
        expected = json.loads(os.environ.get("SQGAN_EXPECTED_RESUME", "null"))
        rates = [[g["lr"] for g in opt.param_groups] for opt in trainer.optimizers]
        actual = {"epoch_index": int(trainer.current_epoch), "global_step": int(trainer.global_step), "optimizer_learning_rates": rates}
        if expected:
            assert actual["global_step"] == expected["global_step"], (actual, expected)
            assert len(trainer.optimizers) == expected["optimizer_states"], (actual, expected)
            assert len(trainer.lr_scheduler_configs) == expected["lr_schedulers"], (actual, expected)
            assert actual["epoch_index"] >= expected["saved_epoch_index"], (actual, expected)
            for actual_groups, expected_groups in zip(rates, expected["optimizer_learning_rates"]):
                assert len(actual_groups) == len(expected_groups)
                assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12) for a, b in zip(actual_groups, expected_groups)), (actual, expected)
            actual["resume_verified"] = True
        else:
            actual["resume_verified"] = False
        if trainer.is_global_zero:
            print("TRAINING STATE VERIFIED", json.dumps(actual), flush=True)
            root = os.environ.get("SQGAN_STAGE_AUDIT_DIR")
            if root:
                destination = Path(root)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "resume_verified.json").write_text(json.dumps(actual, indent=2))

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if trainer.is_global_zero and batch_idx % 100 == 0:
            print("TRAIN PROGRESS", json.dumps({"epoch_index":int(trainer.current_epoch),"batch_index":batch_idx,"global_step":int(trainer.global_step),"training_batches":trainer.num_training_batches}), flush=True)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        elapsed = time.time() - self.epoch_started if self.epoch_started else 0.0
        completed = int(trainer.current_epoch) + 1
        # Leave enough time for checkpoint serialization, mirroring and output upload.
        local_stop = completed >= self.epoch_limit or time.time() + max(600, 1.5 * elapsed) >= self.deadline
        stop = trainer.strategy.reduce_boolean_decision(local_stop, all=False)
        trainer.should_stop = trainer.should_stop or stop
        if trainer.is_global_zero:
            record = {"completed_epochs": completed, "global_step": int(trainer.global_step), "epoch_seconds": elapsed, "session_seconds_remaining": max(0, self.deadline-time.time())}
            for key, value in trainer.callback_metrics.items():
                if getattr(value, "numel", lambda: 0)() == 1:
                    record[key] = float(value.detach().cpu())
            root = os.environ.get("SQGAN_STAGE_AUDIT_DIR")
            if root:
                with (Path(root) / "training_history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            print("EPOCH AUDIT", json.dumps(record), flush=True)
            if stop:
                print("NORMAL SESSION STOP REQUESTED at completed stage epoch", completed, flush=True)


def install():
    if getattr(pl.Trainer, "_sqgan_session_control_installed", False):
        return
    original = pl.Trainer.fit

    def fit(self, *args, **kwargs):
        self.callbacks.append(SessionControl())
        return original(self, *args, **kwargs)

    pl.Trainer.fit = fit
    pl.Trainer._sqgan_session_control_installed = True
