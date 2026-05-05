"""Byte-level tokenizer for early code diffusion experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """A deterministic UTF-8 byte tokenizer with a few special tokens.

    This is deliberately simple: no trained vocabulary, no external files, and
    no tokenization drift between data prep, training, Modal, and inference.
    """

    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")

        payload = list(text.encode("utf-8", errors="replace"))
        payload = payload[: max_length - 2]
        ids = [self.bos_token_id, *payload, self.eos_token_id]
        attention_mask = [1] * len(ids)

        pad = max_length - len(ids)
        if pad > 0:
            ids.extend([self.pad_token_id] * pad)
            attention_mask.extend([0] * pad)

        return ids, attention_mask

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = True) -> str:
        byte_values: list[int] = []
        special = {
            self.pad_token_id,
            self.mask_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        for token_id in ids:
            token_id = int(token_id)
            if token_id in special:
                if skip_special:
                    continue
                byte_values.append(ord("?"))
            elif 0 <= token_id <= 255:
                byte_values.append(token_id)
        return bytes(byte_values).decode("utf-8", errors="replace")


def default_tokenizer() -> ByteTokenizer:
    return ByteTokenizer()
