"""Verify Lightning checkpoint continuation on a tiny CPU model, not SQ-GAN."""
import json
import os
from pathlib import Path

import numpy as np
if not hasattr(np, "Inf"):
    np.Inf = np.inf
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from session_control import install


class Probe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 1)
        self.discriminator = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        layer = self.generator if optimizer_idx == 0 else self.discriminator
        return layer(batch[0]).square().mean()

    def configure_optimizers(self):
        opts = [torch.optim.Adam(layer.parameters(), lr=0.01) for layer in [self.generator, self.discriminator]]
        return opts, [{"scheduler": torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5), "interval": "step"} for opt in opts]


def main(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    install()
    os.environ["SQGAN_SESSION_DEADLINE_UNIX"] = "inf"
    os.environ["SQGAN_STOP_AFTER_STAGE_EPOCHS"] = "2"
    os.environ["SQGAN_EXPECTED_RESUME"] = "null"
    os.environ["SQGAN_STAGE_AUDIT_DIR"] = str(root / "first")
    loader = DataLoader(TensorDataset(torch.ones(8, 2)), batch_size=2)
    def trainer():
        return pl.Trainer(accelerator="cpu", devices=1, max_epochs=4, logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    first = trainer()
    first.fit(Probe(), loader)
    checkpoint = root / "probe.ckpt"
    first.save_checkpoint(checkpoint)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert raw["epoch"] == 2 or raw["epoch"] == 1
    expected = {"saved_epoch_index": raw["epoch"], "global_step": raw["global_step"], "optimizer_states":len(raw["optimizer_states"]),"lr_schedulers":len(raw["lr_schedulers"]),"optimizer_learning_rates":[[g["lr"] for g in opt["param_groups"]] for opt in raw["optimizer_states"]]}
    os.environ["SQGAN_EXPECTED_RESUME"] = json.dumps(expected)
    os.environ["SQGAN_STOP_AFTER_STAGE_EPOCHS"] = "4"
    os.environ["SQGAN_STAGE_AUDIT_DIR"] = str(root / "second")
    second = trainer()
    second.fit(Probe(), loader, ckpt_path=str(checkpoint))
    assert second.current_epoch == 4, second.current_epoch
    assert second.global_step > raw["global_step"]
    os.environ["SQGAN_EXPECTED_RESUME"] = "null"
    os.environ["SQGAN_SESSION_DEADLINE_UNIX"] = "0"
    os.environ["SQGAN_STAGE_AUDIT_DIR"] = str(root / "deadline")
    timed = trainer()
    timed.fit(Probe(), loader)
    assert timed.current_epoch == 1, timed.current_epoch
    timed.save_checkpoint(root / "deadline_stop.ckpt")
    timed_state = torch.load(root / "deadline_stop.ckpt", map_location="cpu", weights_only=False)
    assert len(timed_state["optimizer_states"]) == len(timed_state["lr_schedulers"]) == 2
    report = {"result":"PASS", "initial_global_step":raw["global_step"],"resumed_final_global_step":second.global_step,"resumed_final_epoch":second.current_epoch,"optimizer_and_scheduler_states_restored":True,"deadline_stop_and_checkpoint_verified":True}
    (root / "resume_contract_pass.json").write_text(json.dumps(report, indent=2))
    print("RESUME CONTRACT PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
