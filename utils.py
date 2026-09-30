#!/usr/bin/env python3
"""General utilities for CoR-GraNet training and evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset import ArraySplit, FoldData
from model import CoRGraNet


def load_configs(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def checkpoint_path(root: Path, cohort_cfg: dict, **kwargs) -> Path:
    return root / cohort_cfg["checkpoint_dir"] / cohort_cfg["checkpoint_pattern"].format(**kwargs)


def training_checkpoint_path(root: Path, cohort: str, cohort_cfg: dict, **kwargs) -> Path:
    return root / cohort / cohort_cfg["checkpoint_pattern"].format(**kwargs)


def model_config_for_fold(base_config: dict, fold_data: FoldData) -> dict:
    config = dict(base_config)
    config["structural_node_dim"] = int(fold_data.structural_test.shape[-1])
    config["functional_node_dim"] = int(fold_data.functional_test.shape[-1])
    config["metadata_dim"] = int(
        fold_data.metadata_test.shape[-1] if config.get("raw_metadata_dim", 0) == 0 else config["metadata_dim"]
    )
    config["node_prior_dim"] = int(fold_data.node_prior_test.shape[-1])
    return config


def model_config_for_split(base_config: dict, split: ArraySplit) -> dict:
    config = dict(base_config)
    config["structural_node_dim"] = int(split.structural.shape[-1])
    config["functional_node_dim"] = int(split.functional.shape[-1])
    config["metadata_dim"] = int(split.metadata.shape[-1] if config.get("raw_metadata_dim", 0) == 0 else config["metadata_dim"])
    config["node_prior_dim"] = int(split.node_prior.shape[-1])
    return config


def state_dict_to_cpu(model: CoRGraNet) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_optimizer(model: CoRGraNet, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    return optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)


def parse_fold_filter(spec: str) -> set[int] | None:
    if spec.lower() == "all":
        return None
    return {int(item.strip()) for item in spec.split(",") if item.strip()}


def parse_site_filter(spec: str, available_sites: list[str]) -> list[str]:
    if spec.lower() == "all":
        return available_sites
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available_sites))
    if unknown:
        raise ValueError(f"Unknown ABIDE site(s): {', '.join(unknown)}")
    return requested
