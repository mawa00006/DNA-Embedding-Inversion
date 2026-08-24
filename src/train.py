"""Training and evaluation utilities for DNA sequence reconstruction.

This module provides a standardized `fit` function to orchestrate the training
and evaluation loop for sequence reconstruction tasks. It uses MSE loss to
reconstruct one-hot encoded DNA sequences from embeddings.
"""

from __future__ import annotations

from typing import Dict, Callable, Any
import os
import time
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def _save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_val_loss: float,
    best_state: Dict[str, Any] | None,
    best_epoch: int,
    epochs_without_improvement: int,
    lr_scale: float,
    device: torch.device,
) -> None:
    """Atomically persist full training state so a killed job can resume.

    Captures everything needed to continue training bit-for-bit from the start
    of the next epoch: model/optimizer/scheduler state, early-stopping
    bookkeeping, the best-so-far weights, and RNG state (so the shuffled data
    order continues rather than restarting). Written to a temp file then
    ``os.replace``d into place so a crash mid-write never corrupts the resume
    point.
    """
    tmp_path = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "lr_scale": lr_scale,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    interval: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    tokenizer: Any | None = None,
    grad_clip_norm: float = 0.0,
    lr_scale: float = 1.0,
) -> Dict[str, float]:
    """Perform one full training epoch.

    Sets the model to training mode, iterates over the data loader, computes
    gradients, and updates the model parameters.

    ``lr_scale`` is a multiplier applied on top of the scheduler's LR (the
    adaptive NaN-backoff factor owned by ``fit``); ``1.0`` is a no-op.

    Returns
    -------
    Dict[str, float]
        A dictionary with the average training loss for the epoch.
    """
    model.train()
    losses = []
    skipped_steps = 0

    def _step_scheduler():
        # OneCycleLR overwrites each group's LR from its own step count, so the
        # adaptive backoff multiplier must be re-applied on top after every step
        # for a reduced lr_scale to actually lower the LR the next update sees.
        if scheduler is None:
            return
        scheduler.step()
        if lr_scale != 1.0:
            for group in optimizer.param_groups:
                group["lr"] *= lr_scale

    for batch_idx, (batch_embedding, batch_sequence) in enumerate(loader):
        batch_embedding = batch_embedding.to(device)
        batch_sequence = batch_sequence.to(device)

        # Length targets are derived from the collated (pre-pad-extension) batch_sequence
        # so the count is invariant to any subsequent -100 padding below.
        length_targets = ((batch_sequence != -100).sum(dim=1) - 1).clamp(min=0)

        # Mask-Predict models are conditional masked LMs: the model additionally
        # consumes a partially-masked copy of the target tokens and is trained to
        # fill the masked positions. The mask and the masked-only loss target are
        # produced together (revealed positions are excluded from the loss).
        if hasattr(model, "sample_masked_input"):
            masked_input, seq_target = model.sample_masked_input(batch_sequence)
            output = model(batch_embedding, masked_input)
        else:
            seq_target = batch_sequence
            output = model(batch_embedding)

        # Some models (query_decoder, mask_predict) return (seq_logits, length_logits).
        if isinstance(output, tuple):
            predictions, length_logits = output
        else:
            predictions, length_logits = output, None

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            # Model output: (B, L, V), needs (B, V, L) for CrossEntropyLoss
            # Target: (B, L) indices
            predictions = predictions.permute(0, 2, 1)

            # Pad the target out to the model's output width so every output
            # position is accounted for, with the surplus ignored via -100.
            target_len = seq_target.size(1)
            pred_len = predictions.size(2)

            # A target wider than the model cannot be repaired here: dropping the
            # tail also drops its EOS, so the model would be trained to end the
            # sequence early on exactly the longest samples, and length_targets
            # (computed above, pre-pad) would index past the length head. Both
            # corruptions are silent, so fail instead. train.py sizes
            # effective_seq_length from an exact pass over the data, which is what
            # makes this unreachable rather than merely unlikely.
            assert target_len <= pred_len, (
                f"Target is {target_len} tokens but the model emits {pred_len}: "
                "effective_seq_length was sized too small for this data."
            )
            if target_len < pred_len:
                padding = torch.full(
                    (seq_target.size(0), pred_len - target_len),
                    -100,
                    device=device,
                    dtype=torch.long,
                )
                seq_target = torch.cat([seq_target, padding], dim=1)

        loss = loss_fn(predictions, seq_target)

        if length_logits is not None:
            length_loss = F.cross_entropy(length_logits, length_targets)
            loss = loss + model.aux_length_loss_weight * length_loss

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)

        # A single non-finite loss (a bad batch, or overflow at the OneCycle LR
        # peak) would backprop NaNs into every weight and permanently poison the
        # run -- once the weights are NaN they stay NaN, and every requeue reloads
        # them. Skip the update instead of dying. The LR scheduler is still
        # stepped so the OneCycle schedule stays aligned and completes at
        # total_steps.
        if not torch.isfinite(loss):
            skipped_steps += 1
            _step_scheduler()
            continue

        loss.backward()

        # Clip the global gradient norm before stepping. OneCycle drives the LR
        # up to max_lr (~epoch 3 here); unclipped, the gradients explode and the
        # loss diverges. clip_grad_norm_ returns the *pre-clip* total norm, which
        # we log so spikes are visible.
        grad_norm = None
        if grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # A finite loss can still yield non-finite gradients (overflow in the
        # backward pass); stepping the optimizer with them poisons the weights
        # just the same, so skip here too (grad clipping does not rescue a NaN
        # norm). Scheduler still advances to keep the schedule aligned.
        if grad_norm is not None and not torch.isfinite(grad_norm):
            skipped_steps += 1
            _step_scheduler()
            continue

        optimizer.step()

        # Step scheduler after each batch (required for OneCycleLR)
        _step_scheduler()

        if (batch_idx + 1) % interval == 0:
            log_obj = {
                "batch_idx": batch_idx + 1,
                "loss": loss.item(),
            }
            if grad_norm is not None:
                log_obj["grad_norm"] = float(grad_norm)
            if scheduler is not None:
                log_obj["lr"] = scheduler.get_last_lr()[0]
            log_fn(log_obj)

        losses.append(loss.item())

    # np.mean([]) is nan with a warning: if every batch was skipped the epoch
    # genuinely produced no usable gradient, so report nan explicitly instead.
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "skipped_steps": skipped_steps,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    tokenizer: Any | None = None,
) -> Dict[str, float]:
    """Evaluate the model on a given dataset without computing gradients.

    Sets the model to evaluation mode and computes loss over the provided
    data loader.

    Returns
    -------
    Dict[str, float]
        A dictionary with the average validation loss.
    """
    model.eval()
    losses = []
    for batch_embedding, batch_sequence in loader:
        batch_embedding = batch_embedding.to(device)
        batch_sequence = batch_sequence.to(device)

        length_targets = ((batch_sequence != -100).sum(dim=1) - 1).clamp(min=0)

        # Mirror train_epoch: Mask-Predict models score the masked-LM objective.
        if hasattr(model, "sample_masked_input"):
            masked_input, seq_target = model.sample_masked_input(batch_sequence)
            output = model(batch_embedding, masked_input)
        else:
            seq_target = batch_sequence
            output = model(batch_embedding)

        if isinstance(output, tuple):
            predictions, length_logits = output
        else:
            predictions, length_logits = output, None

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            predictions = predictions.permute(0, 2, 1)

            target_len = seq_target.size(1)
            pred_len = predictions.size(2)

            # Mirrors train_epoch: truncating here would silently score the model
            # against a target stripped of its EOS. See the note there.
            assert target_len <= pred_len, (
                f"Target is {target_len} tokens but the model emits {pred_len}: "
                "effective_seq_length was sized too small for this data."
            )
            if target_len < pred_len:
                padding = torch.full(
                    (seq_target.size(0), pred_len - target_len),
                    -100,
                    device=device,
                    dtype=torch.long,
                )
                seq_target = torch.cat([seq_target, padding], dim=1)

        loss = loss_fn(predictions, seq_target)
        if length_logits is not None:
            length_loss = F.cross_entropy(length_logits, length_targets)
            loss = loss + model.aux_length_loss_weight * length_loss
        losses.append(loss.item())
    return {"loss": float(np.mean(losses))}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    interval: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    early_stopping_enabled: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-6,
    tokenizer: Any | None = None,
    checkpoint_path: str | None = None,
    grad_clip_norm: float = 0.0,
    skip_frac_threshold: float = 1.0,
    lr_backoff: float = 1.0,
) -> Dict[str, float]:
    """Train a model with validation-based early stopping.

    This function implements a complete training loop for sequence reconstruction.
    It monitors validation loss and saves the state of the best-performing model.
    After training is complete, it restores this best state.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer | None
        The optimizer for model parameters. Can be None if epochs=0.
    loss_fn : nn.Module
        Loss function (e.g., MSELoss).
    epochs : int
        Number of training epochs.
    device : torch.device
        Device to run training on (cuda or cpu).
    log_fn : Callable[[Dict[str, Any]], None]
        A function to log metrics (e.g., to console and/or W&B).
    interval : int
        Interval (in batches) at which to log training progress.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None
        Optional learning rate scheduler.
    early_stopping_enabled : bool
        Whether to enable early stopping.
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping.
    early_stopping_min_delta : float
        Minimum change in validation loss to qualify as improvement.
    checkpoint_path : str | None
        If given, full training state is saved here after every epoch and, if
        the file already exists on entry, training resumes from it. This makes
        the loop robust to SLURM time-limit kills: a requeued job continues from
        the last completed epoch instead of restarting from scratch.
    grad_clip_norm : float
        Max global gradient norm; gradients are clipped to this before each
        optimizer step. ``0`` disables clipping. Stabilises the OneCycle LR peak.
    skip_frac_threshold : float
        Adaptive stabiliser trigger. If a non-improving epoch skipped more than
        this fraction of its batches (non-finite loss/grad), the model is rolled
        back to ``best_state`` and the effective LR is multiplied by
        ``lr_backoff``. ``1.0`` effectively disables the trigger.
    lr_backoff : float
        Multiplier applied to the effective LR each time the stabiliser fires.
        ``< 1`` lowers the LR (compounding across triggers, so it cannot loop
        indefinitely); ``1.0`` keeps the rollback but leaves the LR unchanged.
    Returns
    -------
    Dict[str, float]
        A dictionary containing the best validation loss.
    """
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    # Persistent multiplier on top of the scheduler LR, lowered by the adaptive
    # NaN-backoff below and carried across requeues via the checkpoint.
    lr_scale = 1.0

    if epochs == 0:
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        return {
            "best_val_loss": val_metrics["loss"],
            "stopped_early": False,
            "final_epoch": 0,
        }

    # Resume from a previous checkpoint if one exists for this run.
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        # If the last checkpointed weights diverged to NaN/Inf, resuming *from
        # them* trains NaN forever -- the poisoned state (and its poisoned
        # optimizer moments) reloads on every requeue and can never recover. Roll
        # back to the best finite snapshot instead and restart the schedule from
        # that epoch, so the remaining epoch budget actually trains.
        resume_state_finite = all(torch.isfinite(p).all() for p in model.parameters())

        if not resume_state_finite:
            assert ckpt["best_state"] is not None, (
                "Checkpoint weights are non-finite and no best_state is available "
                "to recover from; the run cannot be salvaged automatically."
            )
            # best_epoch tells us which epoch the good weights came from and thus
            # how far to rewind the schedule.
            best_epoch = ckpt["best_epoch"]
            best_val_loss = ckpt["best_val_loss"]
            best_state = ckpt["best_state"]
            model.load_state_dict(best_state)
            # Drop the poisoned optimizer moments by keeping the freshly built
            # optimizer (do not load optimizer_state). RNG is left fresh too, so
            # we do not replay the exact batch order that triggered divergence.
            epochs_without_improvement = 0
            start_epoch = best_epoch + 1
            # Rewind OneCycleLR to the end of best_epoch. The scheduler passed in
            # is freshly constructed with the correct hyperparameters; only its
            # step counter needs moving. After best_epoch epochs it would have
            # stepped best_epoch * len(train_loader) times, so the redone epochs
            # follow the original LR anneal and still land on total_steps at the
            # final epoch (rather than overrunning it and raising).
            if scheduler is not None:
                steps_done = best_epoch * len(train_loader)
                scheduler.last_epoch = steps_done
                scheduler._step_count = steps_done + 1
            log_fn(
                {
                    "message": (
                        f"Recovered from non-finite checkpoint at epoch "
                        f"{ckpt['epoch']}: rolled back to best epoch {best_epoch}, "
                        f"reset optimizer, rewound scheduler; resuming at epoch "
                        f"{start_epoch}"
                    ),
                    "best_val_loss": best_val_loss,
                }
            )
        else:
            if optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if scheduler is not None and ckpt["scheduler_state"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val_loss = ckpt["best_val_loss"]
            best_state = ckpt["best_state"]
            best_epoch = ckpt["best_epoch"]
            # Carry the adaptive backoff factor across requeues.
            lr_scale = ckpt["lr_scale"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            start_epoch = ckpt["epoch"] + 1
            rng = ckpt["rng_state"]
            # torch.load(map_location=device) moves these saved CPU ByteTensors onto
            # the GPU; the RNG setters require CPU ByteTensors, so force them back.
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"] is not None and device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"].cpu(), device)
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            log_fn(
                {
                    "message": f"Resumed from checkpoint at epoch {ckpt['epoch']}; "
                    f"continuing at epoch {start_epoch}",
                    "best_val_loss": best_val_loss,
                }
            )

    # Track the last epoch reached so the return value is well-defined even when
    # a resumed run has no remaining epochs to execute.
    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            log_fn,
            interval,
            scheduler,
            tokenizer,
            grad_clip_norm,
            lr_scale,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, tokenizer)

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "time_sec": elapsed_time,
        }
        # Surface skipped (non-finite) batches so wasted work stays visible in the
        # per-epoch log rather than hiding behind a slightly-off train_loss.
        if train_metrics["skipped_steps"] > 0:
            log_obj["skipped_steps"] = train_metrics["skipped_steps"]
        log_fn(log_obj)

        # Check for improvement with min_delta threshold
        improved = val_metrics["loss"] < best_val_loss - early_stopping_min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Adaptive stabiliser: a non-improving epoch that skipped a large fraction
        # of its batches means the per-batch NaN skip alone is not enough -- the LR
        # is too high for the current state. Roll the (drifted) weights back to the
        # last good snapshot and lower the effective LR so the next epochs train
        # from a known-good point at a calmer LR. Each trigger compounds lr_backoff,
        # so this cannot loop indefinitely.
        num_batches = len(train_loader)
        skip_frac = train_metrics["skipped_steps"] / num_batches if num_batches else 0.0
        if not improved and skip_frac > skip_frac_threshold:
            if best_state is not None:
                model.load_state_dict(best_state)
            lr_scale *= lr_backoff
            log_fn(
                {
                    "message": (
                        f"Adaptive stabiliser fired at epoch {epoch}: "
                        f"skip_frac={skip_frac:.3f} > {skip_frac_threshold}; "
                        f"rolled back to best epoch {best_epoch}, lr_scale={lr_scale:.4g}"
                    ),
                    "best_val_loss": best_val_loss,
                }
            )

        # Persist full training state so a time-limited (SLURM) job can resume
        # from here on requeue instead of restarting from epoch 1.
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                best_state,
                best_epoch,
                epochs_without_improvement,
                lr_scale,
                device,
            )

        # Early stopping check
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            log_obj = {
                "message": f"Early stopping triggered after {epoch} epochs",
                "epochs_without_improvement": epochs_without_improvement,
                "best_val_loss": best_val_loss,
            }
            log_fn(log_obj)
            break

    assert best_state is not None, "Training loop failed to produce a best model state."
    model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "stopped_early": early_stopping_enabled
        and epochs_without_improvement >= early_stopping_patience,
        "final_epoch": epoch,
    }
