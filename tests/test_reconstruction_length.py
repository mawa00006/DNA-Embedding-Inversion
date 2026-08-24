"""Regression tests for variable-length reconstruction output."""

import numpy as np
import torch
from torch import nn

from src.evaluate import reconstruct_sequences
from src.tokenizers import CharacterTokenizer


class _LongPrediction(nn.Module):
    """Return ACGTA + EOS, deliberately longer than the four-nt target hint."""

    def forward(self, inputs: torch.Tensor):
        token_ids = torch.tensor([0, 1, 2, 3, 0, 4], device=inputs.device)
        logits = torch.full((1, len(token_ids), 5), -1_000.0, device=inputs.device)
        logits[0, torch.arange(len(token_ids)), token_ids] = 1_000.0
        length_logits = torch.zeros((1, len(token_ids)), device=inputs.device)
        return logits, length_logits


class _BatchedPrediction(nn.Module):
    """Emit A+EOS or C+EOS according to the first embedding coordinate."""

    def forward(self, inputs: torch.Tensor):
        batch = len(inputs)
        logits = torch.full((batch, 2, 5), -1_000.0, device=inputs.device)
        first = (inputs[:, 0] > 0).long()
        logits[torch.arange(batch), 0, first] = 1_000.0
        logits[:, 1, 4] = 1_000.0
        return logits


def test_mean_decoder_output_is_not_truncated_to_target_length():
    reconstructed = reconstruct_sequences(
        model=_LongPrediction(),
        data={"embeddings": [np.zeros(2, dtype=np.float32)]},
        device=torch.device("cpu"),
        tokenizer=CharacterTokenizer(),
        mode="mean",
        seq_length=4,
        embedding_dim=2,
        data_is_mean=True,
    )

    assert reconstructed == ["ACGTA"]


def test_mean_reconstruction_batches_inputs_without_changing_eos_decoding():
    reconstructed = reconstruct_sequences(
        model=_BatchedPrediction(),
        data={
            "embeddings": [
                np.asarray([-1.0, 0.0], dtype=np.float32),
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([-2.0, 0.0], dtype=np.float32),
            ]
        },
        device=torch.device("cpu"),
        tokenizer=CharacterTokenizer(),
        mode="mean",
        seq_length=4,
        embedding_dim=2,
        data_is_mean=True,
        inference_batch_size=3,
    )
    assert reconstructed == ["A", "C", "A"]
