"""Tokenizer abstractions for DNA sequence models.

This module provides a unified interface for different tokenization strategies:
- Character-level (A, C, G, T, <EOS>)
- HuggingFace AutoTokenizers (BPE, K-mers, etc.)

All tokenizers expose an explicit ``eos_id`` so models can learn to emit a
stop symbol and the decoded sequence length matches the model's prediction.
"""

from __future__ import annotations

import abc
from typing import List, Union
import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils import NUCLEOTIDES


class BaseTokenizer(abc.ABC):
    """Abstract base class for all tokenizers."""

    @abc.abstractproperty
    def vocab_size(self) -> int:
        """Return the size of the vocabulary (including EOS)."""
        pass

    @property
    @abc.abstractmethod
    def eos_id(self) -> int:
        """Return the token id used to mark end-of-sequence."""
        pass

    @abc.abstractmethod
    def encode(self, sequence: str) -> torch.Tensor:
        """Encode a sequence string into a LongTensor of token indices.

        The returned indices do NOT include an EOS token; callers append
        ``eos_id`` themselves once they have applied any length truncation.
        """
        pass

    @abc.abstractmethod
    def token_lengths(self, sequences: List[str]) -> List[int]:
        """Return ``len(encode(s))`` for every sequence, without building tensors.

        Batched so callers can size a decoder from an exact pass over the corpus
        (see ``src.data.max_target_length``); a per-sequence ``encode`` loop is
        an order of magnitude too slow for that.
        """
        pass

    @abc.abstractmethod
    def decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        """Decode token indices into a DNA string, stopping at the first EOS."""
        pass


def _to_list(indices: Union[torch.Tensor, np.ndarray, List[int]]) -> List[int]:
    if isinstance(indices, torch.Tensor):
        return indices.tolist()
    if isinstance(indices, np.ndarray):
        return indices.tolist()
    return list(indices)


class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer for A, C, G, T plus an end-of-sequence symbol.

    Maps:
        A -> 0
        C -> 1
        G -> 2
        T -> 3
        <EOS> -> 4
    """

    EOS_ID = len(NUCLEOTIDES)

    def __init__(self):
        self.nucleotide_map = {n: i for i, n in enumerate(NUCLEOTIDES)}
        self.inverse_map = {i: n for n, i in self.nucleotide_map.items()}

    @property
    def vocab_size(self) -> int:
        return len(NUCLEOTIDES) + 1

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    def encode(self, sequence: str) -> torch.Tensor:
        upper_seq = sequence.upper()
        indices = [self.nucleotide_map[n] for n in upper_seq]
        return torch.tensor(indices, dtype=torch.long)

    def token_lengths(self, sequences: List[str]) -> List[int]:
        # One token per nucleotide, so the length is the string length.
        return [len(s) for s in sequences]

    def decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        idx_list = _to_list(indices)
        chars = []
        for i in idx_list:
            if i == self.EOS_ID:
                break
            chars.append(self.inverse_map[i])
        return "".join(chars)


class HuggingFaceTokenizer(BaseTokenizer):
    """Wrapper around HuggingFace AutoTokenizer.

    Uses the underlying tokenizer's EOS token (falling back to SEP) as the
    explicit end-of-sequence marker. Tokenizers that ship without either
    (e.g. the masked-LM Nucleotide Transformer vocabularies) get a dedicated
    ``<eos>`` special token appended so the decoder still has a stop symbol.
    """

    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.sep_token_id
        if eos is None:
            # No end-of-sequence symbol in the pretrained vocabulary. Register a
            # dedicated one; this only extends the decoder's target vocabulary
            # (embeddings are precomputed and unaffected).
            self.tokenizer.add_special_tokens({"eos_token": "<eos>"})
            eos = self.tokenizer.eos_token_id
        self._eos_id = int(eos)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def encode(self, sequence: str) -> torch.Tensor:
        ids = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.squeeze(0)

    def token_lengths(self, sequences: List[str]) -> List[int]:
        encoded = self.tokenizer(sequences, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        idx_list = _to_list(indices)
        if self._eos_id in idx_list:
            idx_list = idx_list[: idx_list.index(self._eos_id)]
        decoded = self.tokenizer.decode(idx_list, skip_special_tokens=True)
        return decoded.replace(" ", "")
