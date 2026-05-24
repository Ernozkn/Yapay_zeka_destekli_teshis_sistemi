from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SamplePair:
    patient_id: str
    volume_path: Path
    segmentation_path: Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR.parent / "veriler" / "karaciger_3d"


def _extract_id(filename: str, prefix: str) -> str:
    name = filename
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected filename: {filename}")
    return name[len(prefix) :]


def collect_pairs(volumes_dir: Path, segmentations_dir: Path) -> List[SamplePair]:
    volumes_dir = Path(volumes_dir)
    segmentations_dir = Path(segmentations_dir)

    volume_paths = sorted(volumes_dir.glob("volume-*.nii*"))
    pairs: List[SamplePair] = []
    for volume_path in volume_paths:
        patient_id = _extract_id(volume_path.name, "volume-")
        seg_nii = segmentations_dir / f"segmentation-{patient_id}.nii"
        seg_niigz = segmentations_dir / f"segmentation-{patient_id}.nii.gz"
        if seg_nii.exists():
            segmentation_path = seg_nii
        elif seg_niigz.exists():
            segmentation_path = seg_niigz
        else:
            raise FileNotFoundError(
                f"Missing segmentation for {volume_path.name}: {seg_nii} or {seg_niigz}"
            )
        pairs.append(
            SamplePair(
                patient_id=patient_id,
                volume_path=volume_path,
                segmentation_path=segmentation_path,
            )
        )
    if not pairs:
        raise FileNotFoundError(
            f"No volume files found under: {volumes_dir}. Expected volume-*.nii[.gz]"
        )
    return pairs


def split_patient_ids(
    pairs: Sequence[SamplePair], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    patient_ids = sorted({pair.patient_id for pair in pairs})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    if len(patient_ids) <= 1:
        return patient_ids, []
    val_count = max(1, int(round(len(patient_ids) * val_ratio)))
    val_ids = patient_ids[:val_count]
    train_ids = patient_ids[val_count:]
    return train_ids, val_ids


def _load_nifti(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} from {path}")
    return data


def _clip_and_normalize(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32, copy=False)


def _random_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    rng: np.random.Generator,
    pos_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    pd, ph, pw = patch_size
    d, h, w = volume.shape

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        pad_width = (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )
        volume = np.pad(volume, pad_width, mode="constant", constant_values=0)
        mask = np.pad(mask, pad_width, mode="constant", constant_values=0)
        d, h, w = volume.shape

    start_d = int(rng.integers(0, d - pd + 1))
    start_h = int(rng.integers(0, h - ph + 1))
    start_w = int(rng.integers(0, w - pw + 1))

    if pos_ratio > 0 and rng.random() < pos_ratio:
        tumor_coords = np.argwhere(mask == 2)
        if tumor_coords.size > 0:
            coord = tumor_coords[int(rng.integers(0, len(tumor_coords)))]
            center_d, center_h, center_w = int(coord[0]), int(coord[1]), int(coord[2])
            start_d = min(max(center_d - pd // 2, 0), d - pd)
            start_h = min(max(center_h - ph // 2, 0), h - ph)
            start_w = min(max(center_w - pw // 2, 0), w - pw)

    volume_patch = volume[
        start_d : start_d + pd,
        start_h : start_h + ph,
        start_w : start_w + pw,
    ]
    mask_patch = mask[
        start_d : start_d + pd,
        start_h : start_h + ph,
        start_w : start_w + pw,
    ]
    return volume_patch, mask_patch


def _random_augment(
    volume: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    axes_pairs = ((0, 1), (0, 2), (1, 2))
    if rng.random() < 0.5:
        axes = axes_pairs[int(rng.integers(0, len(axes_pairs)))]
        k = int(rng.integers(1, 4))
        volume = np.rot90(volume, k=k, axes=axes)
        mask = np.rot90(mask, k=k, axes=axes)

    for axis in range(3):
        if rng.random() < 0.5:
            volume = np.flip(volume, axis=axis)
            mask = np.flip(mask, axis=axis)

    return np.ascontiguousarray(volume), np.ascontiguousarray(mask)


def _worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed = worker_info.seed % 2**32
    np.random.seed(seed)


class LiverTumorDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[SamplePair],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        hu_min: float = -100.0,
        hu_max: float = 200.0,
        pos_ratio: float = 0.5,
        samples_per_volume: int = 4,
        augment: bool = True,
        seed: int | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.patch_size = patch_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.pos_ratio = pos_ratio
        self.samples_per_volume = max(1, int(samples_per_volume))
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_volume

    def _get_rng(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = worker_info.seed if worker_info else 0
        return np.random.default_rng(self.seed + worker_seed + index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_index = index // self.samples_per_volume
        pair = self.pairs[pair_index]
        volume = _load_nifti(pair.volume_path)
        mask = _load_nifti(pair.segmentation_path)

        volume = _clip_and_normalize(volume, self.hu_min, self.hu_max)
        mask = mask.astype(np.int64, copy=False)

        rng = self._get_rng(index)
        volume_patch, mask_patch = _random_patch(
            volume,
            mask,
            self.patch_size,
            rng,
            pos_ratio=self.pos_ratio,
        )

        if self.augment:
            volume_patch, mask_patch = _random_augment(volume_patch, mask_patch, rng)

        volume_tensor = torch.from_numpy(volume_patch).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch)
        return volume_tensor, mask_tensor


def build_datasets(
    data_root: Path | str | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    hu_min: float = -100.0,
    hu_max: float = 200.0,
    pos_ratio: float = 0.5,
    samples_per_volume: int = 4,
) -> Tuple[LiverTumorDataset, LiverTumorDataset]:
    data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    volumes_dir = data_root / "volumes"
    segmentations_dir = data_root / "segmentations"

    pairs = collect_pairs(volumes_dir, segmentations_dir)
    train_ids, val_ids = split_patient_ids(pairs, val_ratio=val_ratio, seed=seed)

    train_pairs = [pair for pair in pairs if pair.patient_id in train_ids]
    val_pairs = [pair for pair in pairs if pair.patient_id in val_ids]

    train_ds = LiverTumorDataset(
        train_pairs,
        patch_size=patch_size,
        hu_min=hu_min,
        hu_max=hu_max,
        pos_ratio=pos_ratio,
        samples_per_volume=samples_per_volume,
        augment=True,
        seed=seed,
    )
    val_ds = LiverTumorDataset(
        val_pairs,
        patch_size=patch_size,
        hu_min=hu_min,
        hu_max=hu_max,
        pos_ratio=pos_ratio,
        samples_per_volume=samples_per_volume,
        augment=False,
        seed=seed + 1,
    )
    return train_ds, val_ds


def build_dataloaders(
    data_root: Path | str | None = None,
    batch_size: int = 1,
    num_workers: int = 2,
    val_ratio: float = 0.2,
    seed: int = 42,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    hu_min: float = -100.0,
    hu_max: float = 200.0,
    pos_ratio: float = 0.5,
    samples_per_volume: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(
        data_root=data_root,
        val_ratio=val_ratio,
        seed=seed,
        patch_size=patch_size,
        hu_min=hu_min,
        hu_max=hu_max,
        pos_ratio=pos_ratio,
        samples_per_volume=samples_per_volume,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )
    return train_loader, val_loader
