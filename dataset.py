#!/usr/bin/env python3
"""Dataset utilities used to reproduce the CoR-GraNet Table 1 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset


@dataclass
class FoldData:
    structural_test: np.ndarray
    functional_test: np.ndarray
    adjacency_test: np.ndarray
    node_prior_test: np.ndarray
    metadata_test: np.ndarray
    labels_test: np.ndarray


@dataclass
class ArraySplit:
    structural: np.ndarray
    functional: np.ndarray
    adjacency: np.ndarray
    node_prior: np.ndarray
    metadata: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class TrainFoldData:
    train: ArraySplit
    validation: ArraySplit | None
    test: ArraySplit


@dataclass
class SiteDataset:
    site: str
    subjects: list[str]
    labels: np.ndarray
    structural_nodes: np.ndarray
    functional_nodes: np.ndarray
    adjacency: np.ndarray
    metadata: np.ndarray
    node_prior: np.ndarray | None = None
    metadata_keep_raw_indices: tuple[int, ...] = ()


class GraphFeatureDataset(Dataset):
    """PyTorch Dataset wrapper for prepared graph-feature arrays."""

    def __init__(
        self,
        structural: np.ndarray,
        functional: np.ndarray,
        adjacency: np.ndarray,
        node_prior: np.ndarray,
        metadata: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.structural = torch.as_tensor(structural, dtype=torch.float32)
        self.functional = torch.as_tensor(functional, dtype=torch.float32)
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.node_prior = torch.as_tensor(node_prior, dtype=torch.float32)
        self.metadata = torch.as_tensor(metadata, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    @classmethod
    def from_split(cls, split: ArraySplit) -> "GraphFeatureDataset":
        return cls(
            split.structural,
            split.functional,
            split.adjacency,
            split.node_prior,
            split.metadata,
            split.labels,
        )

    @classmethod
    def from_fold_data(cls, fold_data: FoldData) -> "GraphFeatureDataset":
        return cls(
            fold_data.structural_test,
            fold_data.functional_test,
            fold_data.adjacency_test,
            fold_data.node_prior_test,
            fold_data.metadata_test,
            fold_data.labels_test,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.structural[index],
            self.functional[index],
            self.adjacency[index],
            self.node_prior[index],
            self.metadata[index],
            self.labels[index],
        )


def standardize_fold(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    keep = np.isfinite(flat).all(axis=0) & (flat.std(axis=0) > 0.0)
    train_x = train_x[..., keep]
    test_x = test_x[..., keep]
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    train_x = ((train_x - mean) / np.clip(std, 1e-6, None)).astype(np.float32)
    test_x = ((test_x - mean) / np.clip(std, 1e-6, None)).astype(np.float32)
    return train_x, test_x


def standardize_metadata_keep_dims(
    train_x: np.ndarray,
    test_x: np.ndarray,
    keep_raw_indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    if train_x.shape[1] == 0:
        return train_x.astype(np.float32), test_x.astype(np.float32)
    train_x = train_x.astype(np.float32).copy()
    test_x = test_x.astype(np.float32).copy()
    keep_raw = set(keep_raw_indices)
    for i in range(train_x.shape[1]):
        train_col = train_x[:, i]
        test_col = test_x[:, i]
        finite = np.isfinite(train_col)
        fill = float(np.nanmedian(train_col[finite])) if finite.any() else 0.0
        train_col[~finite] = fill
        test_col[~np.isfinite(test_col)] = fill
        if i in keep_raw:
            train_x[:, i] = train_col
            test_x[:, i] = test_col
            continue
        mean = train_col.mean()
        std = train_col.std()
        train_x[:, i] = (train_col - mean) / max(std, 1e-6)
        test_x[:, i] = (test_col - mean) / max(std, 1e-6)
    return train_x.astype(np.float32), test_x.astype(np.float32)


def load_openneuro_metadata(tsv_path: Path, subjects: np.ndarray) -> np.ndarray:
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing OpenNeuro participants.tsv: {tsv_path}")
    frame = pd.read_csv(tsv_path, sep="\t")
    id_col = "participant_id" if "participant_id" in frame.columns else frame.columns[0]
    frame["sub_clean"] = frame[id_col].astype(str).str.replace("sub-", "", regex=False)
    sub_clean = [str(subject).replace("sub-", "") for subject in subjects]
    aligned = frame.set_index("sub_clean").reindex(sub_clean)
    age = (
        aligned["age"].to_numpy(dtype=np.float32)
        if "age" in aligned.columns
        else np.zeros(len(subjects), dtype=np.float32)
    )
    iq = (
        aligned["intelligence_quotient"].to_numpy(dtype=np.float32)
        if "intelligence_quotient" in aligned.columns
        else np.zeros(len(subjects), dtype=np.float32)
    )
    for values in (age, iq):
        finite = np.isfinite(values)
        values[~finite] = float(np.nanmedian(values[finite])) if finite.any() else 0.0
    hand_right = (
        (aligned["hand"].fillna("R") == "R").to_numpy(dtype=np.float32)
        if "hand" in aligned.columns
        else np.ones(len(subjects), dtype=np.float32)
    )
    metadata = np.stack([age, iq, hand_right, age**2, iq**2, age * iq], axis=1)
    return metadata.astype(np.float32)


def load_openneuro_npz(dataset_npz: Path, participants_tsv: Path) -> SiteDataset:
    data = np.load(dataset_npz, allow_pickle=True)
    subjects = data["participant_ids"]
    node_prior = data.get(
        "node_prior",
        data.get(
            "node_meg_prior_features",
            np.zeros((len(data["labels"]), data["mri_node_features"].shape[1], 0), dtype=np.float32),
        ),
    )
    metadata = load_openneuro_metadata(participants_tsv, subjects)
    return SiteDataset(
        site="OpenNeuro",
        subjects=[str(subject) for subject in subjects],
        labels=data["labels"].astype(np.int64),
        structural_nodes=data["mri_node_features"].astype(np.float32),
        functional_nodes=data["meg_node_features"].astype(np.float32),
        adjacency=np.abs(data["adjacency"]).astype(np.float32),
        metadata=metadata,
        node_prior=node_prior.astype(np.float32),
        metadata_keep_raw_indices=(),
    )


def build_openneuro_fold(dataset: SiteDataset, test_idx: np.ndarray) -> FoldData:
    node_prior = dataset.node_prior
    if node_prior is None:
        node_prior = np.zeros(
            (len(dataset.labels), dataset.structural_nodes.shape[1], 0),
            dtype=np.float32,
        )
    return FoldData(
        structural_test=dataset.structural_nodes[test_idx].astype(np.float32),
        functional_test=dataset.functional_nodes[test_idx].astype(np.float32),
        adjacency_test=dataset.adjacency[test_idx].astype(np.float32),
        node_prior_test=node_prior[test_idx].astype(np.float32),
        metadata_test=dataset.metadata[test_idx].astype(np.float32),
        labels_test=dataset.labels[test_idx].astype(np.int64),
    )


def openneuro_fold_from_npz(
    dataset_npz: Path,
    participants_tsv: Path,
    test_idx: np.ndarray,
) -> FoldData:
    data = np.load(dataset_npz, allow_pickle=True)
    subjects = data["participant_ids"]
    metadata = load_openneuro_metadata(participants_tsv, subjects)
    node_prior = data.get(
        "node_prior",
        data.get(
            "node_meg_prior_features",
            np.zeros((len(data["labels"]), data["mri_node_features"].shape[1], 0), dtype=np.float32),
        ),
    )
    return FoldData(
        structural_test=data["mri_node_features"][test_idx].astype(np.float32),
        functional_test=data["meg_node_features"][test_idx].astype(np.float32),
        adjacency_test=np.abs(data["adjacency"][test_idx]).astype(np.float32),
        node_prior_test=node_prior[test_idx].astype(np.float32),
        metadata_test=metadata[test_idx].astype(np.float32),
        labels_test=data["labels"][test_idx].astype(np.int64),
    )


def load_abide_cached_site(cache_dir: Path, site: str) -> SiteDataset:
    path = cache_dir / f"{site}_age6_12_difumo128_global_fs6_extended_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing ABIDE cached features: {path}")
    data = np.load(path, allow_pickle=True)
    nodes = data["nodes"].astype(np.float32)
    meg_node_dim = 36
    structural_nodes = nodes[:, :, :-meg_node_dim]
    functional_nodes = nodes[:, :, -meg_node_dim:]
    return SiteDataset(
        site=site,
        subjects=[str(subject) for subject in data["subjects"]],
        labels=data["labels"].astype(np.int64),
        structural_nodes=structural_nodes.astype(np.float32),
        functional_nodes=functional_nodes.astype(np.float32),
        adjacency=data["adj"].astype(np.float32),
        metadata=data["meta"].astype(np.float32),
        metadata_keep_raw_indices=(6, 7, 8),
    )


def build_abide_fold(dataset: SiteDataset, train_idx: np.ndarray, test_idx: np.ndarray) -> FoldData:
    structural_train, structural_test = standardize_fold(
        dataset.structural_nodes[train_idx],
        dataset.structural_nodes[test_idx],
    )
    functional_train, functional_test = standardize_fold(
        dataset.functional_nodes[train_idx],
        dataset.functional_nodes[test_idx],
    )
    del structural_train, functional_train
    metadata_train, metadata_test = standardize_metadata_keep_dims(
        dataset.metadata[train_idx],
        dataset.metadata[test_idx],
        keep_raw_indices=dataset.metadata_keep_raw_indices,
    )
    del metadata_train
    return FoldData(
        structural_test=structural_test,
        functional_test=functional_test,
        adjacency_test=dataset.adjacency[test_idx].astype(np.float32),
        node_prior_test=np.zeros(
            (len(test_idx), dataset.structural_nodes.shape[1], 0),
            dtype=np.float32,
        ),
        metadata_test=metadata_test,
        labels_test=dataset.labels[test_idx].astype(np.int64),
    )


def subset_split(split: ArraySplit, indices: np.ndarray) -> ArraySplit:
    return ArraySplit(
        structural=split.structural[indices],
        functional=split.functional[indices],
        adjacency=split.adjacency[indices],
        node_prior=split.node_prior[indices],
        metadata=split.metadata[indices],
        labels=split.labels[indices],
    )


def make_validation_split(labels: np.ndarray, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    train_indices = np.arange(len(labels))
    if val_size <= 0.0 or len(np.unique(labels)) < 2:
        return train_indices, None
    if np.bincount(labels.astype(np.int64), minlength=2).min() < 2:
        return train_indices, None
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        return next(splitter.split(np.zeros(len(labels)), labels))
    except ValueError:
        return train_indices, None


def split_train_validation(split: ArraySplit, val_size: float, seed: int) -> tuple[ArraySplit, ArraySplit | None]:
    train_idx, val_idx = make_validation_split(split.labels, val_size, seed)
    if val_idx is None:
        return split, None
    return subset_split(split, train_idx), subset_split(split, val_idx)


def build_openneuro_training_fold(
    dataset: SiteDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_size: float,
    seed: int,
) -> TrainFoldData:
    node_prior = dataset.node_prior
    if node_prior is None:
        node_prior = np.zeros((len(dataset.labels), dataset.structural_nodes.shape[1], 0), dtype=np.float32)
    train_full = ArraySplit(
        structural=dataset.structural_nodes[train_idx].astype(np.float32),
        functional=dataset.functional_nodes[train_idx].astype(np.float32),
        adjacency=dataset.adjacency[train_idx].astype(np.float32),
        node_prior=node_prior[train_idx].astype(np.float32),
        metadata=dataset.metadata[train_idx].astype(np.float32),
        labels=dataset.labels[train_idx].astype(np.int64),
    )
    train, validation = split_train_validation(train_full, val_size, seed)
    test = ArraySplit(
        structural=dataset.structural_nodes[test_idx].astype(np.float32),
        functional=dataset.functional_nodes[test_idx].astype(np.float32),
        adjacency=dataset.adjacency[test_idx].astype(np.float32),
        node_prior=node_prior[test_idx].astype(np.float32),
        metadata=dataset.metadata[test_idx].astype(np.float32),
        labels=dataset.labels[test_idx].astype(np.int64),
    )
    return TrainFoldData(train=train, validation=validation, test=test)


def build_abide_training_fold(
    dataset: SiteDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_size: float,
    seed: int,
) -> TrainFoldData:
    structural_train, structural_test = standardize_fold(
        dataset.structural_nodes[train_idx],
        dataset.structural_nodes[test_idx],
    )
    functional_train, functional_test = standardize_fold(
        dataset.functional_nodes[train_idx],
        dataset.functional_nodes[test_idx],
    )
    metadata_train, metadata_test = standardize_metadata_keep_dims(
        dataset.metadata[train_idx],
        dataset.metadata[test_idx],
        keep_raw_indices=dataset.metadata_keep_raw_indices,
    )
    node_prior_train = np.zeros((len(train_idx), structural_train.shape[1], 0), dtype=np.float32)
    node_prior_test = np.zeros((len(test_idx), structural_test.shape[1], 0), dtype=np.float32)
    train_full = ArraySplit(
        structural=structural_train,
        functional=functional_train,
        adjacency=dataset.adjacency[train_idx].astype(np.float32),
        node_prior=node_prior_train,
        metadata=metadata_train,
        labels=dataset.labels[train_idx].astype(np.int64),
    )
    train, validation = split_train_validation(train_full, val_size, seed)
    test = ArraySplit(
        structural=structural_test,
        functional=functional_test,
        adjacency=dataset.adjacency[test_idx].astype(np.float32),
        node_prior=node_prior_test,
        metadata=metadata_test,
        labels=dataset.labels[test_idx].astype(np.int64),
    )
    return TrainFoldData(train=train, validation=validation, test=test)
