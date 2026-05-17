"""Tokenizers for code diffusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class CodeTokenizer(Protocol):
    pad_token_id: int
    mask_token_id: int
    bos_token_id: int
    eos_token_id: int
    fim_prefix_token_id: int
    fim_suffix_token_id: int
    fim_middle_token_id: int

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]: ...

    def encode_payload(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = True) -> str: ...


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
    fim_prefix_token_id: int = 260
    fim_suffix_token_id: int = 261
    fim_middle_token_id: int = 262

    @property
    def vocab_size(self) -> int:
        return 263

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")

        payload = self.encode_payload(text)
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
            self.fim_prefix_token_id,
            self.fim_suffix_token_id,
            self.fim_middle_token_id,
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

    def encode_payload(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))


class BPETokenizer:
    """Wrapper around a Hugging Face `tokenizers` BPE tokenizer JSON."""

    special_tokens = {
        "pad": "<pad>",
        "mask": "<mask>",
        "bos": "<bos>",
        "eos": "<eos>",
        "fim_prefix": "<fim_prefix>",
        "fim_suffix": "<fim_suffix>",
        "fim_middle": "<fim_middle>",
        "unk": "<unk>",
    }

    def __init__(self, path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install tokenizers to use BPETokenizer") from exc
        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.pad_token_id = self._id("pad")
        self.mask_token_id = self._id("mask")
        self.bos_token_id = self._id("bos")
        self.eos_token_id = self._id("eos")
        self.fim_prefix_token_id = self._id("fim_prefix")
        self.fim_suffix_token_id = self._id("fim_suffix")
        self.fim_middle_token_id = self._id("fim_middle")

    def _id(self, name: str) -> int:
        token_id = self.tokenizer.token_to_id(self.special_tokens[name])
        if token_id is None:
            raise ValueError(f"Tokenizer is missing special token {self.special_tokens[name]}")
        return int(token_id)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode_payload(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text).ids)

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")
        payload = self.encode_payload(text)[: max_length - 2]
        ids = [self.bos_token_id, *payload, self.eos_token_id]
        attention_mask = [1] * len(ids)
        pad = max_length - len(ids)
        if pad > 0:
            ids.extend([self.pad_token_id] * pad)
            attention_mask.extend([0] * pad)
        return ids, attention_mask

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = True) -> str:
        if skip_special:
            special = {
                self.pad_token_id,
                self.mask_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.fim_prefix_token_id,
                self.fim_suffix_token_id,
                self.fim_middle_token_id,
            }
            ids = [int(token_id) for token_id in ids if int(token_id) not in special]
        return self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special)


def train_bpe_tokenizer(texts: Iterable[str], output_path: str | Path, vocab_size: int = 16384) -> BPETokenizer:
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install tokenizers to train a BPE tokenizer") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token=BPETokenizer.special_tokens["unk"]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special_tokens = [
        BPETokenizer.special_tokens["pad"],
        BPETokenizer.special_tokens["mask"],
        BPETokenizer.special_tokens["bos"],
        BPETokenizer.special_tokens["eos"],
        BPETokenizer.special_tokens["fim_prefix"],
        BPETokenizer.special_tokens["fim_suffix"],
        BPETokenizer.special_tokens["fim_middle"],
        BPETokenizer.special_tokens["unk"],
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(output_path))
    return BPETokenizer(output_path)


def load_tokenizer_from_config(config: dict) -> CodeTokenizer:
    tokenizer_cfg = config.get("tokenizer", {"type": "byte"})
    tokenizer_type = tokenizer_cfg.get("type", "byte")
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "bpe":
        return BPETokenizer(tokenizer_cfg["path"])
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def default_tokenizer() -> ByteTokenizer:
    return ByteTokenizer()
