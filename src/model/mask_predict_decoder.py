"""Mask-Predict (CMLM) refiner for inverting mean-pooled DNA embeddings.

This is the iterative-refinement counterpart to ``query_decoder``. The
single-shot decoders predict every output position independently in one forward
pass, so at longer sequence lengths their per-position predictions are
conditionally independent given the embedding and become mutually incoherent.
Mask-Predict fixes this by conditioning each refinement pass on the tokens
already decided.

The model is trained as a conditional masked language model (CMLM): it consumes
the mean embedding *and* a partially-masked copy of the target tokens, and is
trained to fill the masked positions. Crucially, the random-mask-ratio training
objective already covers every partial-observation state the iterative decoder
visits, so training is a single forward/backward per batch -- the inference loop
is *not* unrolled during training.

Architecture (mean mode only), forked from ``QueryDecoderReconstructor``:

1. ``context_projection`` maps the mean embedding to ``n_context_tokens``
   learnable summary tokens of size ``d_model`` (cross-attention memory).
2. ``length_head`` predicts a distribution over content lengths from the mean
   embedding; the soft distribution is mixed with ``length_embedding`` and added
   to every position query, exactly as in the query decoder.
3. ``token_embedding`` embeds the current draft token at each position. A
   dedicated ``MASK`` symbol (index ``output_dim``, input-only) marks
   not-yet-decided positions. This is the one addition over the query decoder:
   it lets the model see its own partial reconstruction.
4. ``pos_queries`` are learnable per-slot embeddings. Combined with the length
   conditioning and the per-position token embedding, they form the decoder
   inputs.
5. A stack of ``TransformerDecoderLayer`` blocks cross-attends queries -> context
   and self-attends queries <-> queries.
6. ``output_projection`` produces per-position vocabulary logits.

At inference, :meth:`iterative_decode` sizes a canvas from the length head,
starts fully masked (so the first pass equals the single-shot prediction), and
repeatedly keeps the most-confident positions while re-masking and re-predicting
the rest. ``num_iterations == 1`` reproduces the single-shot decoder exactly.

Forward returns a tuple ``(seq_logits, length_logits)``; training combines the
masked-position sequence loss with a cross-entropy loss on the length head
(weighted by ``aux_length_loss_weight``).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MaskPredictReconstructor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mode: str,
        seq_length: int,
        output_dim: int,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        n_context_tokens: int = 16,
        aux_length_loss_weight: float = 0.5,
        num_iterations: int = 10,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        assert input_dim > 0, "input_dim must be positive"
        assert mode == "mean", "MaskPredictReconstructor supports mean mode only"
        assert seq_length > 0, "seq_length must be positive"
        assert output_dim > 0, "output_dim must be positive"
        assert d_model > 0, "d_model must be positive"
        assert nhead > 0 and d_model % nhead == 0, "nhead must divide d_model"
        assert num_layers > 0, "num_layers must be positive"
        assert dim_feedforward > 0, "dim_feedforward must be positive"
        assert 0.0 <= dropout <= 1.0, "dropout in [0, 1]"
        assert n_context_tokens > 0, "n_context_tokens must be positive"
        assert aux_length_loss_weight >= 0.0, "aux_length_loss_weight must be non-negative"
        assert num_iterations > 0, "num_iterations must be positive"

        self.input_dim = input_dim
        self.mode = mode
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_context_tokens = n_context_tokens
        self.aux_length_loss_weight = aux_length_loss_weight
        self.num_iterations = num_iterations
        # Input-only MASK symbol, placed just past the output vocabulary so it
        # never collides with a real class. The output projection still emits
        # only `output_dim` classes.
        self.mask_id = output_dim

        logger = logging.getLogger(__name__)
        logger.info(
            f"MaskPredict: input_dim={input_dim} -> {n_context_tokens} context tokens "
            f"x d_model={d_model}, seq_length={seq_length} (incl. EOS slot), "
            f"output_dim={output_dim} (mask_id={self.mask_id}), num_layers={num_layers}, "
            f"nhead={nhead}, aux_length_loss_weight={aux_length_loss_weight}, "
            f"num_iterations={num_iterations}"
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
        # +1 for the input-only MASK symbol.
        self.token_embedding = nn.Embedding(output_dim + 1, d_model)

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

    def forward(
        self, x: torch.Tensor, y_in: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one masked-LM pass.

        Parameters
        ----------
        x : torch.Tensor
            Mean-pooled embeddings, shape (B, input_dim).
        y_in : torch.Tensor | None
            Per-position input token ids, shape (B, seq_length), where
            not-yet-decided positions hold ``mask_id``. If ``None`` the canvas is
            fully masked, which reproduces the single-shot prediction.

        Returns
        -------
        seq_logits : torch.Tensor
            Per-position vocabulary logits, shape (B, seq_length, output_dim).
        length_logits : torch.Tensor
            Per-sample length logits, shape (B, seq_length).
        """
        assert x.ndim == 2, f"Expected 2D mean embedding, got {x.ndim}D"
        B, D = x.shape
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"

        if y_in is None:
            y_in = torch.full(
                (B, self.seq_length), self.mask_id, dtype=torch.long, device=x.device
            )
        assert y_in.shape == (B, self.seq_length), (
            f"Expected y_in of shape {(B, self.seq_length)}, got {tuple(y_in.shape)}"
        )

        context = self.context_projection(x).view(B, self.n_context_tokens, self.d_model)

        length_logits = self.length_head(x)
        length_probs = F.softmax(length_logits, dim=-1)
        length_emb = length_probs @ self.length_embedding.weight

        queries = self.pos_queries.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + length_emb.unsqueeze(1) + self.token_embedding(y_in)

        hidden = self.transformer_decoder(tgt=queries, memory=context)
        seq_logits = self.output_projection(hidden)

        return seq_logits, length_logits

    def sample_masked_input(
        self, batch_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a randomly-masked input and its CMLM loss target.

        Each collated row is ``[content..., EOS, -100, -100, ...]``. A random
        number of the real (non-pad) positions is replaced with ``mask_id`` in
        the input; the loss target keeps the true ids only at those masked
        positions (revealed and pad positions are ``-100``), so cross-entropy is
        computed on the masked positions alone (standard CMLM).

        Both returned tensors are sized to the model canvas ``seq_length`` so
        they align with the model output without further padding.

        Returns
        -------
        y_in : torch.Tensor
            Input token ids with masks, shape (B, seq_length).
        loss_target : torch.Tensor
            Masked-only targets, shape (B, seq_length), ``-100`` elsewhere.
        """
        B = batch_sequence.size(0)
        L = self.seq_length
        device = batch_sequence.device

        # Place the collated targets onto the fixed canvas.
        target = torch.full((B, L), -100, dtype=torch.long, device=device)
        t = min(batch_sequence.size(1), L)
        target[:, :t] = batch_sequence[:, :t]

        real = target != -100  # content + EOS positions
        real_counts = real.sum(dim=1)

        # Number of real positions to mask, in [1, real_count] (inclusive of the
        # fully-masked case so inference step 0 is trained for).
        rand = torch.rand(B, device=device)
        num_mask = (rand * real_counts.float()).floor().long() + 1
        num_mask = torch.minimum(num_mask, real_counts.clamp(min=1))

        # Mask the highest-scored `num_mask` real positions (random scores). Pad
        # positions get score -1 so they always rank last and are never chosen.
        scores = torch.rand(B, L, device=device).masked_fill(~real, -1.0)
        order = scores.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(L, device=device).unsqueeze(0).expand(B, L))
        masked = rank < num_mask.unsqueeze(1)

        y_in = target.clone()
        y_in[~real] = self.mask_id  # post-EOS / pad positions are "masked" input
        y_in[masked] = self.mask_id

        loss_target = torch.full((B, L), -100, dtype=torch.long, device=device)
        loss_target[masked] = target[masked]

        return y_in, loss_target

    @torch.no_grad()
    def iterative_decode(
        self, x: torch.Tensor, eos_id: int, num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        """Decode sequences by iterative mask-predict refinement.

        Sizes a per-sample canvas from the length head, starts fully masked, and
        for ``num_iterations`` steps keeps the most-confident positions while
        re-masking and re-predicting the rest (confidence is recomputed each
        step, so early mistakes can be revised). Positions beyond the predicted
        length are filled with ``eos_id`` so the caller's decoder terminates
        cleanly and never sees ``mask_id``.

        Parameters
        ----------
        x : torch.Tensor
            Mean-pooled embeddings, shape (B, input_dim).
        eos_id : int
            End-of-sequence id used to fill positions past the predicted length.
        num_iterations : int | None
            Number of refinement passes; defaults to ``self.num_iterations``.
            ``1`` reproduces the single-shot decoder.

        Returns
        -------
        torch.Tensor
            Decoded token ids, shape (B, seq_length).
        """
        T = self.num_iterations if num_iterations is None else num_iterations
        assert T > 0, "num_iterations must be positive"
        assert x.ndim == 2, f"Expected 2D mean embedding, got {x.ndim}D"
        B = x.size(0)
        L = self.seq_length
        device = x.device

        length_logits = self.length_head(x)
        content_len = length_logits.argmax(dim=-1)  # number of content tokens
        canvas = (content_len + 1).clamp(min=1, max=L)  # + EOS slot

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        active = pos < canvas.unsqueeze(1)

        y = torch.full((B, L), self.mask_id, dtype=torch.long, device=device)
        for step in range(1, T + 1):
            logits, _ = self.forward(x, y)
            conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
            conf = conf.masked_fill(~active, -1.0)  # never keep inactive slots

            n_keep = torch.floor(canvas.float() * step / T).long().clamp(min=1)
            n_keep = torch.minimum(n_keep, canvas)

            order = conf.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, pos)
            keep = rank < n_keep.unsqueeze(1)

            y = torch.where(keep, pred, torch.full_like(y, self.mask_id))

        # At the final step n_keep == canvas, so every active slot is filled.
        y = torch.where(active, y, torch.full_like(y, eos_id))
        return y
