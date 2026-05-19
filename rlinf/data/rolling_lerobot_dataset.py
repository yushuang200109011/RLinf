# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import bisect
import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rlinf.data.utils import build_dataloader_from_dataset
from rlinf.utils.logging import get_logger

logger = get_logger()

_META_KEYS: frozenset[str] = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)

def _hf_column_to_numpy_bool_1d(hf_dataset: Any, col: str) -> np.ndarray:
    raw = hf_dataset[col]
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.stack(list(raw))
    return t.reshape(-1).bool().numpy()


def _hf_column_to_numpy_int64_1d(hf_dataset: Any, col: str) -> np.ndarray:
    raw = hf_dataset[col]
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.stack(list(raw))
    return t.reshape(-1).to(torch.int64).numpy()


# ---------------------------------------------------------------------------
# In-memory Arrow store (used by RollingLeRobotDataset.in_memory_mode)
# ---------------------------------------------------------------------------


class InMemoryArrowStore:
    def __init__(
        self,
        chunk_size: int = 1,
        action_sequence_keys: list[str] | None = None,
        fps: int = 10,
        image_transforms: Callable | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._fps = max(fps, 1)
        self._image_transforms = image_transforms
        keys = action_sequence_keys or []
        self._delta_indices: dict[str, list[int]] = (
            {k: list(range(chunk_size)) for k in keys} if chunk_size > 1 else {}
        )
        self._episode_datasets: deque[Any] = deque()
        self._ep_from: deque[int] = deque()
        self._ep_to: deque[int] = deque()
        self._total_frames: int = 0
        self._hf_features: Any = None
        self._image_keys: set[str] = set()
        self._task_to_idx: dict[str, int] = {}
        self._tasks: dict[int, str] = {}
        self._hits: int = 0

    # ------------------------------------------------------------------
    # Schema inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_hf_features(frame: dict) -> tuple[Any, set[str]]:
        import datasets as hf_datasets

        meta_features: dict[str, Any] = {
            "index": hf_datasets.Value("int64"),
            "episode_index": hf_datasets.Value("int64"),
            "frame_index": hf_datasets.Value("int64"),
            "timestamp": hf_datasets.Value("float32"),
            "task_index": hf_datasets.Value("int64"),
        }
        data_features: dict[str, Any] = {}
        image_keys: set[str] = set()

        for key, val in frame.items():
            if key == "task":
                continue
            if not isinstance(val, np.ndarray):
                continue
            if val.dtype == np.uint8 and val.ndim == 3:
                data_features[key] = hf_datasets.Image()
                image_keys.add(key)
            elif val.shape == (1,):
                dtype_str = (
                    "bool" if np.issubdtype(val.dtype, np.bool_) else str(val.dtype)
                )
                data_features[key] = hf_datasets.Value(dtype_str)
            elif val.ndim == 1:
                dtype_str = (
                    "bool" if np.issubdtype(val.dtype, np.bool_) else str(val.dtype)
                )
                data_features[key] = hf_datasets.Sequence(
                    length=val.shape[0],
                    feature=hf_datasets.Value(dtype_str),
                )
            elif val.ndim == 2:
                data_features[key] = hf_datasets.Array2D(
                    shape=tuple(val.shape), dtype=str(val.dtype)
                )

        return hf_datasets.Features({**meta_features, **data_features}), image_keys

    # ------------------------------------------------------------------
    # Episode addition
    # ------------------------------------------------------------------

    def _prepare_episode_dataset(
        self,
        ep_frames: list[dict],
        base: int | None = None,
        episode_index: int | None = None,
    ) -> dict[str, Any]:
        import PIL.Image as PILImage
        from datasets import Dataset
        from lerobot.common.datasets.utils import hf_transform_to_torch

        n = len(ep_frames)
        if n == 0:
            return {}

        hf_features = self._hf_features
        image_keys = self._image_keys
        if hf_features is None:
            hf_features, image_keys = self._infer_hf_features(ep_frames[0])

        ep_idx = len(self._ep_from) if episode_index is None else int(episode_index)
        base = self._total_frames if base is None else int(base)
        task_to_idx = dict(self._task_to_idx)
        tasks = dict(self._tasks)

        # Register task strings and build per-frame task_index array.
        task_indices: list[int] = []
        for frame in ep_frames:
            task_str = frame.get("task", "")
            if task_str not in task_to_idx:
                tidx = len(task_to_idx)
                task_to_idx[task_str] = tidx
                tasks[tidx] = task_str
            task_indices.append(task_to_idx[task_str])

        ep_dict: dict[str, Any] = {
            "index": np.arange(base, base + n, dtype=np.int64),
            "episode_index": np.full((n,), ep_idx, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "timestamp": np.arange(n, dtype=np.float32) / self._fps,
            "task_index": np.array(task_indices, dtype=np.int64),
        }

        for key in hf_features:
            if key in (
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
            ):
                continue
            if key in image_keys:
                ep_dict[key] = [PILImage.fromarray(f[key]) for f in ep_frames]
            else:
                vals = [f.get(key) for f in ep_frames]
                if all(v is not None for v in vals):
                    stacked = np.stack(vals)
                    # Scalar (1,) — squeeze to 1-D so Arrow Value dtype matches.
                    ep_dict[key] = (
                        stacked.squeeze(1) if stacked.shape == (n, 1) else stacked
                    )

        ep_ds = Dataset.from_dict(ep_dict, features=hf_features)
        ep_ds.set_transform(hf_transform_to_torch)
        return {
            "dataset": ep_ds,
            "base": base,
            "num_frames": n,
            "hf_features": hf_features,
            "image_keys": image_keys,
            "task_to_idx": task_to_idx,
            "tasks": tasks,
        }

    def _commit_prepared_episode(self, prepared_episode: dict[str, Any]) -> int:
        if not prepared_episode:
            return 0

        base = int(prepared_episode["base"])
        if base != self._total_frames:
            raise ValueError("Prepared episode base no longer matches store length.")

        n = int(prepared_episode["num_frames"])
        self._hf_features = prepared_episode["hf_features"]
        self._image_keys = prepared_episode["image_keys"]
        self._task_to_idx = prepared_episode["task_to_idx"]
        self._tasks = prepared_episode["tasks"]
        self._episode_datasets.append(prepared_episode["dataset"])
        self._ep_from.append(base)
        self._ep_to.append(base + n)
        self._total_frames += n
        return n

    def add_episode(self, ep_frames: list[dict]) -> None:
        prepared_episode = self._prepare_episode_dataset(ep_frames)
        if not prepared_episode:
            return
        self._commit_prepared_episode(prepared_episode)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes stored in this shard."""
        return len(self._episode_datasets)

    def __getitem__(self, local_idx: int) -> dict[str, Any]:
        local_idx = max(0, min(local_idx, self._total_frames - 1))
        self._hits += 1
        ep_idx = bisect.bisect_right(self._ep_to, local_idx)
        ep_start = self._ep_from[ep_idx]
        ep_end = self._ep_to[ep_idx]
        local_frame = local_idx - ep_start

        ep_ds = self._episode_datasets[ep_idx]
        item: dict[str, Any] = ep_ds[local_frame]
        item["task"] = self._tasks.get(int(item["task_index"].item()), "")

        if self._image_transforms is not None:
            for k in self._image_keys:
                if k in item:
                    item[k] = self._image_transforms(item[k])

        if self._delta_indices:
            query_indices = {
                key: [max(ep_start, min(ep_end - 1, local_idx + d)) for d in deltas]
                for key, deltas in self._delta_indices.items()
            }
            padding = {
                f"{key}_is_pad": torch.BoolTensor(
                    [
                        (local_idx + d < ep_start) | (local_idx + d >= ep_end)
                        for d in deltas
                    ]
                )
                for key, deltas in self._delta_indices.items()
            }
            # All chunk indices are within the same episode — use local offsets.
            query_result = {
                key: torch.stack(ep_ds.select([q - ep_start for q in q_idxs])[key])
                for key, q_idxs in query_indices.items()
                if key in self._hf_features and key not in self._image_keys
            }
            item = {**item, **padding}
            for k, v in query_result.items():
                item[k] = v

        return item

    def stats(self) -> dict[str, Any]:
        return {
            "in_memory_store_episodes": len(self._episode_datasets),
            "in_memory_store_frames": self._total_frames,
            "in_memory_store_hits": self._hits,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RollingLeRobotDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        chunk_size: int = 1,
        delta_timestamps: dict[str, list[float]] | None = None,
        keys: list[str] | None = None,
        image_transforms: Callable | None = None,
        min_frames: int = 10,
        wait_interval_s: float = 10.0,
        action_sequence_keys: list[str] | None = ["actions"],
        # check intervene
        require_all_intervene: bool = False,
        intervene_flag_key: str = "intervene_flag",
        window_size: int | None = None,
        index_load_workers: int = 1,
        in_memory_mode: bool = False,
        fps: int = 10,
    ) -> None:
        if not in_memory_mode:
            raise ValueError(
                "RollingLeRobotDataset now supports only in_memory_mode=True. "
                "Archived LeRobot shards are not used for training."
            )

        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.keys = keys
        self.image_transforms = image_transforms
        self.min_frames = min_frames
        self.wait_interval_s = wait_interval_s
        self.action_sequence_keys = action_sequence_keys
        self.require_all_intervene = bool(require_all_intervene)
        self.intervene_flag_key = intervene_flag_key
        self.window_size = window_size
        self._window_physical_start: int = 0
        self._window_valid_slice_lo: int = 0
        self._valid_physical_indices: list[int] | None = None
        self._valid_physical_set: set[int] | None = None
        if self.require_all_intervene:
            self._valid_physical_indices = []
            self._valid_physical_set = set()
        # Serializes in-memory index growth vs __getitem__/__getitems__.
        self._rolling_access_lock = threading.RLock()

        # Sub-dataset **roots** indexed so far (paths only; no live LeRobot handles).
        self._sub_datasets: list[Path] = []

        # Prefix-sum of lengths for O(log n) index dispatch.
        # _cumulative_lengths[i] = sum of lengths of sub_datasets[0..i-1].
        self._cumulative_lengths: list[int] = [0]

        # Running total of episodes across all loaded sub-datasets.
        self._total_episodes: int = 0

        self._shard_cache_enabled: bool = True
        self._in_memory_shards: dict[Path, InMemoryArrowStore] = {}
        self._shard_cache_chunk_size: int = chunk_size
        self._shard_cache_fps: int = max(1, int(fps))
        self._shard_cache_action_keys: list[str] = list(action_sequence_keys or [])
        self._shard_cache_hits: int = 0
        self._shard_cache_misses: int = 0

    # ------------------------------------------------------------------
    # Readiness gate
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return len(self) >= self.min_frames

    def _num_physical_frames(self) -> int:
        """Total indexed frames (ignores intervene filter and window)."""
        return int(self._cumulative_lengths[-1])

    def _logical_to_physical(self, logical_idx: int) -> int:
        if self._valid_physical_indices is None:
            if self._window_enabled():
                return self._window_physical_start + int(logical_idx)
            return int(logical_idx)
        if self._window_enabled():
            lo = self._window_valid_slice_lo
            return int(self._valid_physical_indices[lo + int(logical_idx)])
        return int(self._valid_physical_indices[logical_idx])

    def _window_enabled(self) -> bool:
        return self.window_size is not None and int(self.window_size) > 0

    def _update_window_sampling_bounds(self) -> None:
        n = self._num_physical_frames()
        if not self._window_enabled():
            self._window_physical_start = 0
            self._window_valid_slice_lo = 0
            return
        w = max(0, int(self.window_size))
        if self._valid_physical_indices is not None:
            self._window_valid_slice_lo = max(0, len(self._valid_physical_indices) - w)
            if self._window_valid_slice_lo < len(self._valid_physical_indices):
                self._window_physical_start = self._valid_physical_indices[
                    self._window_valid_slice_lo
                ]
            else:
                self._window_physical_start = n
        else:
            self._window_physical_start = max(0, n - w)
            self._window_valid_slice_lo = 0

    def _load_item_from_lerobot(self, idx: int) -> dict[str, Any]:
        ds_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[ds_idx]
        path = self._sub_datasets[ds_idx]
        store = self._in_memory_shards.get(path)
        if store is not None:
            self._shard_cache_hits += 1
            item = store[local_idx]
            if self.keys is not None:
                item = {k: v for k, v in item.items() if k in self.keys}
            return item
        self._shard_cache_misses += 1
        raise RuntimeError(
            f"Indexed in-memory LeRobot shard is missing: {path}. "
            "Training does not fall back to archived disk shards."
        )

    # ------------------------------------------------------------------
    # Shard cache API
    # ------------------------------------------------------------------

    def _new_in_memory_store(self) -> InMemoryArrowStore:
        return InMemoryArrowStore(
            chunk_size=self._shard_cache_chunk_size,
            action_sequence_keys=self._shard_cache_action_keys,
            fps=self._shard_cache_fps,
            image_transforms=self.image_transforms,
        )

    def append_episode_to_memory(self, path: str | Path, ep_frames: list[dict]) -> None:
        if not ep_frames:
            return
        path = Path(path)
        with self._rolling_access_lock:
            if path in self._in_memory_shards:
                ds_idx = self._sub_datasets.index(path)
                if ds_idx != len(self._sub_datasets) - 1:
                    raise ValueError(
                        "Can only append to the latest in-memory LeRobot shard."
                    )
                store = self._in_memory_shards[path]
            else:
                store = self._new_in_memory_store()

            old_len = len(store)
            episode_index = store.num_episodes

        # Build the expensive HuggingFace episode outside the sampling lock.
        prepared_episode = store._prepare_episode_dataset(
            ep_frames,
            base=old_len,
            episode_index=episode_index,
        )
        if not prepared_episode:
            return

        with self._rolling_access_lock:
            if path in self._in_memory_shards:
                ds_idx = self._sub_datasets.index(path)
                if ds_idx != len(self._sub_datasets) - 1:
                    raise ValueError(
                        "Can only append to the latest in-memory LeRobot shard."
                    )
                store = self._in_memory_shards[path]
                physical_base = self._cumulative_lengths[ds_idx]
            else:
                self._in_memory_shards[path] = store
                self._sub_datasets.append(path)
                physical_base = self._cumulative_lengths[-1]
                self._cumulative_lengths.append(physical_base)

            old_len = int(prepared_episode["base"])
            added_frames = store._commit_prepared_episode(prepared_episode)
            if added_frames <= 0:
                return

            self._cumulative_lengths[-1] += added_frames
            self._total_episodes += 1
            if self.require_all_intervene and self._valid_physical_indices is not None:
                assert self._valid_physical_set is not None
                ep_ds = prepared_episode["dataset"]
                if self.intervene_flag_key not in ep_ds.column_names:
                    logger.warning(
                        "[RollingLeRobotDataset] require_all_intervene=True but "
                        "column %r missing in appended in-memory episode; keeping "
                        "all %d chunk starts for this episode.",
                        self.intervene_flag_key,
                        added_frames,
                    )
                    valid_locals = range(old_len, old_len + added_frames)
                else:
                    flags = _hf_column_to_numpy_bool_1d(ep_ds, self.intervene_flag_key)
                    assert int(flags.shape[0]) == added_frames
                    deltas = np.arange(max(1, int(self.chunk_size)), dtype=np.int64)
                    offsets = np.arange(added_frames, dtype=np.int64)[:, None]
                    raw = offsets + deltas[None, :]
                    in_episode = raw < added_frames
                    step_ok = np.ones_like(in_episode, dtype=np.bool_)
                    step_ok[in_episode] = flags[raw[in_episode]]
                    valid_offsets = np.nonzero(step_ok.all(axis=1))[0]
                    valid_locals = (old_len + int(offset) for offset in valid_offsets)
                for local_i in valid_locals:
                    gidx = physical_base + int(local_i)
                    self._valid_physical_indices.append(gidx)
                    self._valid_physical_set.add(gidx)

            self._update_window_sampling_bounds()
            self._evict_stale_shards()
            logger.debug(
                "[RollingLeRobotDataset] episode appended: %s "
                "(+%d frames, physical_frames=%d, logical_samples=%d)",
                path.name,
                added_frames,
                self._num_physical_frames(),
                len(self),
            )

    def _evict_stale_shards(self) -> int:
        if not self._in_memory_shards or not self._window_enabled():
            return 0
        n_evicted = 0
        for ds_idx, path in enumerate(self._sub_datasets):
            shard_end = self._cumulative_lengths[ds_idx + 1]
            if shard_end <= self._window_physical_start:
                if self._in_memory_shards.pop(path, None) is not None:
                    n_evicted += 1
        if n_evicted:
            logger.debug(
                "[RollingLeRobotDataset] evicted %d shard(s) from shard cache "
                "(window_physical_start=%d)",
                n_evicted,
                self._window_physical_start,
            )
        return n_evicted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _refresh_impl(self, new_paths: list[Path]) -> int:
        return 0

    def refresh(self) -> int:
        return 0

    def refresh_one(self) -> int:
        return 0

    def _total_logical_samples(self) -> int:
        """Total logical samples across all shards, ignoring ``window_size``."""
        if self._valid_physical_indices is not None:
            return len(self._valid_physical_indices)
        return self._num_physical_frames()

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "num_sub_datasets": len(self._sub_datasets),
            "physical_frames": self._num_physical_frames(),
            "total_logical_samples": self._total_logical_samples(),
            "logical_samples": len(self),
            "total_frames": len(self),
            "total_episodes": self._total_episodes,
            "require_all_intervene": self.require_all_intervene,
            "window_size": self.window_size if self.window_size is not None else 0,
            "window_physical_start": self._window_physical_start,
            "shard_cache_enabled": self._shard_cache_enabled,
            "shard_cache_shards": len(self._in_memory_shards),
            "shard_cache_hits": self._shard_cache_hits,
            "shard_cache_misses": self._shard_cache_misses,
        }
        return stats

    def __len__(self) -> int:
        if self._valid_physical_indices is not None:
            if self._window_enabled():
                return len(self._valid_physical_indices) - self._window_valid_slice_lo
            return len(self._valid_physical_indices)
        n = int(self._cumulative_lengths[-1])
        if self._window_enabled():
            return max(0, n - self._window_physical_start)
        return n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with self._rolling_access_lock:
            physical = self._logical_to_physical(int(idx))
            return self._load_item_from_lerobot(physical)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Batch fetch for DataLoader (one call per batch when supported)."""
        if not indices:
            return []
        with self._rolling_access_lock:
            physicals = [self._logical_to_physical(int(i)) for i in indices]
            return [self._load_item_from_lerobot(p) for p in physicals]


# ---------------------------------------------------------------------------
# Preload wrapper
# ---------------------------------------------------------------------------


class PreloadRollingLeRobotDataset:
    def __init__(
        self,
        dataset: RollingLeRobotDataset,
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        prefetch_size: int = 5,
        use_random_replacement: bool = True,
        num_samples_per_epoch: int | None = None,
        seed: int = 42,
        num_workers: int = 4,
        **dataloader_kwargs: Any,
    ) -> None:
        assert prefetch_size > 0, f"{prefetch_size=} must be greater than 0"

        self.dataset = dataset
        # Cache all DataLoader construction kwargs for rebuild on refresh().
        self._dl_kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            use_random_replacement=use_random_replacement,
            num_samples_per_epoch=num_samples_per_epoch,
            seed=seed,
            num_workers=num_workers,
            **dataloader_kwargs,
        )
        self.prefetch_size = prefetch_size

        self._stop_event = threading.Event()
        # Guards swaps of self._loader so the background thread sees a
        # consistent reference when refresh() installs a new DataLoader.
        self._loader_lock = threading.Lock()
        self._bg_epoch: int = 0
        self.preload_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=prefetch_size
        )
        self.sample_thread: threading.Thread | None = None
        self._exception: Exception | None = None

        self._loader: DataLoader = self._build_loader()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_loader(self) -> DataLoader:
        """Build a fresh DataLoader using the cached construction kwargs."""
        return build_dataloader_from_dataset(
            dataset=self.dataset,
            **self._dl_kwargs,
        )

    def _sample_worker(self) -> None:
        current_loader: DataLoader | None = None
        loader_iter = None

        while not self._stop_event.is_set():
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            if not self.dataset.is_ready():
                time.sleep(3)
                continue

            # Pick up a rebuilt loader installed by refresh(), if any.
            with self._loader_lock:
                if self._loader is not current_loader:
                    current_loader = self._loader
                    loader_iter = None  # reset iterator for the new loader

            if loader_iter is None:
                if hasattr(current_loader.sampler, "set_epoch"):
                    current_loader.sampler.set_epoch(self._bg_epoch)
                loader_iter = iter(current_loader)

            try:
                batch = next(loader_iter)
            except StopIteration:
                # One DataLoader epoch exhausted: advance and restart.
                self._bg_epoch += 1
                if hasattr(current_loader.sampler, "set_epoch"):
                    current_loader.sampler.set_epoch(self._bg_epoch)
                loader_iter = iter(current_loader)
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    time.sleep(1)
                    continue
            except Exception as e:  # noqa: BLE001
                logger.error("[PreloadRollingLeRobotDataset] sampling error: %s", e)
                self._exception = e
                self._stop_event.set()
                break

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of batches per epoch — mirrors the internal DataLoader."""
        return len(self._loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_worker, daemon=True
            )
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    if self._exception is not None:
                        raise RuntimeError(
                            "Sampling thread failed"
                        ) from self._exception
                    break
                continue

    def refresh(self) -> int:
        n_new = self.dataset.refresh()
        if n_new > 0:
            with self._loader_lock:
                self._loader = self._build_loader()
        return n_new

    def refresh_one(self) -> int:
        n_new = self.dataset.refresh_one()
        if n_new > 0:
            with self._loader_lock:
                self._loader = self._build_loader()
        return n_new

    def close(self) -> None:
        """Stop the background thread and release resources."""
        self._stop_event.set()
        thread_timeout = 10
        if self.sample_thread is not None and self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    "[PreloadRollingLeRobotDataset] sample thread still alive "
                    "after %d seconds, force killing",
                    thread_timeout,
                )

    def __del__(self) -> None:
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_rolling_lerobot_dataset(
    root_dir: str | Path,
    chunk_size: int = 1,
    delta_timestamps: dict[str, list[float]] | None = None,
    keys: list[str] | None = None,
    image_transforms: Callable | None = None,
    min_frames: int = 1,
    wait_interval_s: float = 10.0,
    action_sequence_keys: list[str] | None = ["actions"],
    require_all_intervene: bool = False,
    intervene_flag_key: str = "intervene_flag",
    window_size: int | None = None,
    index_load_workers: int = 1,
    in_memory_mode: bool = False,
    fps: int = 10,
) -> RollingLeRobotDataset:
    dataset = RollingLeRobotDataset(
        root_dir=root_dir,
        chunk_size=chunk_size,
        delta_timestamps=delta_timestamps,
        keys=keys,
        image_transforms=image_transforms,
        min_frames=min_frames,
        wait_interval_s=wait_interval_s,
        action_sequence_keys=action_sequence_keys,
        require_all_intervene=require_all_intervene,
        intervene_flag_key=intervene_flag_key,
        window_size=window_size,
        index_load_workers=index_load_workers,
        in_memory_mode=in_memory_mode,
        fps=fps,
    )

    logger.info(
        "[build_rolling_lerobot_dataset] root_dir=%s, chunk_size=%d, "
        "sub_datasets=%d, logical_samples=%d, "
        "physical_frames=%d, require_all_intervene=%s, window_size=%s",
        root_dir,
        chunk_size,
        len(dataset._sub_datasets),
        len(dataset),
        dataset._num_physical_frames(),
        require_all_intervene,
        window_size,
    )

    return dataset
