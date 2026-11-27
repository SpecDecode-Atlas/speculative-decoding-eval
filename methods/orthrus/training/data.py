"""Arrow rollout loading and document-isolated, reproducible one-epoch packing."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def load_records(path):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
    path = Path(path)
    if path.is_file() and path.suffix == ".arrow":
        dataset = Dataset.from_file(str(path))
    elif (path / "state.json").is_file() or (path / "dataset_dict.json").is_file():
        dataset = load_from_disk(str(path))
    elif (path / "train" / "state.json").is_file():
        dataset = load_from_disk(str(path / "train"))
    else:
        shards = sorted(path.glob("*.arrow"))
        if not shards or any("indices" in p.name or p.name.startswith(("test", "validation")) for p in shards):
            raise ValueError(f"Expected a saved Hugging Face training dataset or a directory of training Arrow shards at {path}")
        dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in shards])
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"Expected pretokenized Qwen rollouts. Found columns {dataset.column_names}")
    return dataset


def _bounds(n, attention, row):
    if attention is None:
        return 0, n
    mask = np.asarray(attention)
    if mask.shape != (n,) or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"Invalid attention_mask at row {row}")
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0, 0
    if valid[-1] - valid[0] + 1 != len(valid):
        raise ValueError(f"Internal gap in attention_mask at row {row}")
    return int(valid[0]), int(valid[-1]) + 1


def _scan(data, max_len):
    import pyarrow.compute as pc
    fields = [name for name in ("input_ids", "attention_mask", "loss_mask", "labels") if name in data.column_names]
    indices, lengths = [], []
    offset = 0
    for batch in data.select_columns(fields).with_format("arrow").iter(batch_size=512):
        sizes = pc.list_value_length(batch["input_ids"]).to_pylist()
        for j, n in enumerate(sizes):
            index = offset + j
            if n is None:
                raise ValueError(f"Null input_ids at row {index}")
            def get_values(column):
                if column not in fields:
                    return None
                item = batch[column][j]
                if not item.is_valid or item.values.null_count:
                    raise ValueError(f"Null {column} at row {index}")
                return item.values.to_numpy(zero_copy_only=False)
            start, stop = _bounds(n, get_values("attention_mask"), index)
            stop = min(stop, start + max_len)
            if stop - start < 2:
                continue
            mask = get_values("loss_mask")
            if mask is None and "labels" in fields:
                mask = get_values("labels") != -100
            if mask is not None:
                if len(mask) != n or not np.isfinite(mask).all() or (mask < 0).any():
                    raise ValueError(f"Invalid supervision mask at row {index}")
                if not np.any(mask[start + 1:stop] > 0):
                    continue
            indices.append(index)
            lengths.append(stop - start)
        offset += len(sizes)
    return np.asarray(indices, dtype=np.int64), np.asarray(lengths, dtype=np.int32)


def lengths_index(data, path, max_len, cache_dir):
    from filelock import FileLock
    key = json.dumps([str(Path(path).resolve()), data._fingerprint, len(data), max_len, "orthrus-index-v1"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache = cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".npz")
    with FileLock(str(cache) + ".lock"):
        if not cache.exists():
            rows, lengths = _scan(data, max_len)
            tmp = cache.with_suffix(f".{os.getpid()}.tmp")
            with tmp.open("wb") as f:
                np.savez(f, rows=rows, lengths=lengths)
            os.replace(tmp, cache)
    with np.load(cache, allow_pickle=False) as f:
        return f["rows"], f["lengths"]


class Documents:
    def __init__(self, data, max_len, smoke=False):
        self.data, self.max_len, self.smoke = data, max_len, smoke

    def __getitem__(self, index):
        row = self.data[int(index)]
        raw = np.asarray(row["input_ids"])
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"Expected integer input_ids at row {index}")
        start, stop = _bounds(len(raw), row.get("attention_mask"), index)
        stop = min(stop, start + self.max_len)
        if row.get("loss_mask") is not None:
            values = np.asarray(row["loss_mask"], dtype=np.float32)
        elif row.get("labels") is not None:
            values = (np.asarray(row["labels"]) != -100).astype(np.float32)
        else:
            values = np.ones(raw.shape, dtype=np.float32)
        if self.smoke and stop - start > 64:
            hits = np.flatnonzero(values[start + 1:stop] > 0)
            if len(hits):
                start = max(start, min(start + int(hits[0]), stop - 64))
            stop = min(start + 64, stop)
        # Keep worker output as ordinary NumPy arrays. Only a few KiB travel
        # per row and no shared Torch-storage file descriptors are needed.
        return {"row": int(index), "ids": raw[start:stop].astype(np.int64),
                "weights": values[start:stop].copy()}


def local_plan(rows, lengths, budget, world, rank, seed):
    """Use the same upstream Multipack implementation as the Ascend DFlash recipe."""
    from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
    sampler = MultipackDistributedBatchSamplerV2(budget, lengths, world, rank, seed=seed)
    sampler.set_epoch(0)
    return [rows[indices].tolist() for indices in sampler]


def append_tail(plan, rows, used, world, rank, seed):
    # The upstream sampler consumes a prefix of this permutation and drops <world rows.
    remaining = len(rows) - used
    if remaining < 0 or remaining >= world:
        raise RuntimeError(f"Unexpected native Multipack tail: {remaining} rows")
    if remaining:
        tail = np.random.default_rng(seed).permutation(len(rows))[used:]
        plan.append([int(rows[tail[rank]])] if rank < remaining else [])


def sample_blocks(rows, max_anchors, seed):
    """Sample starts across the full pack, weighted by number of eligible starts per document."""
    starts = [torch.nonzero(row["weights"][1:] > 0, as_tuple=True)[0].numpy() for row in rows]
    endpoints = np.cumsum([len(s) for s in starts])
    total = int(endpoints[-1]) if len(endpoints) else 0
    chosen = np.sort(np.random.default_rng(seed).choice(total, min(total, max_anchors), replace=False))
    result, offset = [], 0
    for positions, end in zip(starts, endpoints):
        inside = chosen[(chosen >= offset) & (chosen < end)] - offset
        result.append(torch.from_numpy(positions[inside].astype(np.int64)))
        offset = int(end)
    return result


def slot_weights(row, anchors, proposals=15, decay=7.0):
    """Return finite per-slot weights; mask proposals that extend beyond a document."""
    offsets = torch.arange(1, proposals + 1)
    labels = anchors[:, None] + offsets
    present = labels < len(row["ids"])
    return row["weights"][labels.clamp_max(len(row["ids"]) - 1)] * present * torch.exp(-(offsets.float() - 1) / decay)


def identity_collate(rows):
    return rows


def tensor_rows(rows):
    return [{"row": row["row"], "ids": torch.from_numpy(row["ids"]),
             "weights": torch.from_numpy(row["weights"])} for row in rows]
