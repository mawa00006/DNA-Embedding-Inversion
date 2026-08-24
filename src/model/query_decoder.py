"""Query-based decoder for inverting mean-pooled DNA embeddings.

Designed for the multi-length inversion-attack setting: a single fixed-size
mean embedding must be decoded into a variable-length nucleotide sequence
whose length itself is unknown.

Architecture (mean mode only):

1. ``context_projection`` maps the mean embedding to ``n_context_tokens``
   learnable summary tokens of size ``d_model``. These are the "memory" that
   cross-attention attends to.
2. ``length_head`` predicts a distribution over possible content lengths
   (0..seq_length-1) directly from the mean embedding.
3. The soft length distribution is mixed with ``length_embedding`` to produce
   a single conditioning vector that is added to every position query. This
   makes length prediction a first-class input to content generation, not a
   side branch: gradients flow from the sequence loss back through the length
   head, and the decoder is trained to produce EOS at the predicted position.
4. ``pos_queries`` are learnable per-output-slot embeddings. Combined with the
   length conditioning, they form the decoder inputs.
5. A stack of ``TransformerDecoderLayer`` blocks performs cross-attention
   (queries -> context) and self-attention (queries <-> queries).
6. ``output_projection`` produces per-position logits over the vocabulary
   (which includes the EOS symbol).

Forward returns a tuple ``(seq_logits, length_logits)``. The training loop
combines the sequence loss with a cross-entropy loss on the length head
(weighted by ``aux_length_loss_weight``). At inference the length head's
output is consumed only internally and is not exposed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class QueryDecoderReconstructor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mode: str,
        seq_length: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_context_tokens: int = 8,
        aux_length_loss_weight: float = 0.5,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        assert input_dim > 0, "input_dim must be positive"
        assert mode == "mean", "QueryDecoderReconstructor supports mean mode only"
        assert seq_length > 0, "seq_length must be positive"
        assert output_dim > 0, "output_dim must be positive"
        assert d_model > 0, "d_model must be positive"
        assert nhead > 0 and d_model % nhead == 0, "nhead must divide d_model"
        assert num_layers > 0, "num_layers must be positive"
        assert dim_feedforward > 0, "dim_feedforward must be positive"
        assert 0.0 <= dropout <= 1.0, "dropout in [0, 1]"
        assert n_context_tokens > 0, "n_context_tokens must be positive"
        assert aux_length_loss_weight >= 0.0, "aux_length_loss_weight must be non-negative"

        self.input_dim = input_dim
        self.mode = mode
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_context_tokens = n_context_tokens
        self.aux_length_loss_weight = aux_length_loss_weight

        logger = logging.getLogger(__name__)
        logger.info(
            f"QueryDecoder: input_dim={input_dim} -> {n_context_tokens} context tokens "
            f"x d_model={d_model}, seq_length={seq_length} (incl. EOS slot), "
            f"output_dim={output_dim}, num_layers={num_layers}, nhead={nhead}, "
            f"aux_length_loss_weight={aux_length_loss_weight}"
        )

        self.context_projection = nn.Linear(input_dim, n_context_tokens * d_model)

        self.length_head = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, seq_length),
        )

        self.length_embedding = nn.Embedding(seq_length, d_model)
        self.pos_queries = nn.Embedding(seq_length, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, output_dim)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model.

        Parameters
        ----------
        x : torch.Tensor
            Mean-pooled embeddings, shape (B, input_dim).

        Returns
        -------
        seq_logits : torch.Tensor
            Per-position vocabulary logits, shape (B, seq_length, output_dim).
        length_logits : torch.Tensor
            Per-sample length logits, shape (B, seq_length). The class index
            equals the number of content tokens before the EOS slot.
        """
        assert x.ndim == 2, f"Expected 2D mean embedding, got {x.ndim}D"
        B, D = x.shape
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"

        context = self.context_projection(x).view(B, self.n_context_tokens, self.d_model)

        length_logits = self.length_head(x)
        length_probs = F.softmax(length_logits, dim=-1)
        length_emb = length_probs @ self.length_embedding.weight

        queries = self.pos_queries.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + length_emb.unsqueeze(1)

        hidden = self.transformer_decoder(tgt=queries, memory=context)
        seq_logits = self.output_projection(hidden)

        return seq_logits, length_logits

    @staticmethod
    def length_targets_from_padded(batch_sequence: torch.Tensor) -> torch.Tensor:
        """Derive integer length targets from collated padded sequences.

        Each padded row is ``[content..., EOS, -100, -100, ...]``. The target
        we want the length head to predict is the count of content tokens
        before EOS, i.e. (non-pad count) - 1, clamped at 0 for the empty case.
        """
        non_pad = (batch_sequence != -100).sum(dim=1)
        return (non_pad - 1).clamp(min=0)
