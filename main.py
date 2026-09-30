#!/usr/bin/env python3
"""Train or evaluate CoR-GraNet on prepared graph-feature inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset import (
    ArraySplit,
    GraphFeatureDataset,
    TrainFoldData,
    build_abide_fold,
    build_abide_training_fold,
    build_openneuro_training_fold,
    load_abide_cached_site,
    load_openneuro_npz,
    openneuro_fold_from_npz,
)
from metrics import metrics_from_predictions, summarize_folds, weighted_site_summary
from model import CoRGraNet
from utils import (
    build_optimizer,
    checkpoint_path,
    load_configs,
    model_config_for_fold,
    model_config_for_split,
    parse_fold_filter,
    parse_site_filter,
    set_seed,
    state_dict_to_cpu,
    training_checkpoint_path,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def apply_training_profile_values(args: argparse.Namespace, values: dict | None) -> None:
    if not values:
        return
    key_map = {
        "learning_rate": "lr",
        "feature_mask_rate": "feat_mask_rate",
        "early_stopping_patience": "patience",
    }
    ignored_keys = {
        "name",
        "selection_rule",
        "n",
        "losses",
        "lr_schedule",
        "dropout",
    }
    for key, value in values.items():
        if key in ignored_keys:
            continue
        arg_key = key_map.get(key, key)
        if not hasattr(args, arg_key):
            continue
        if arg_key == "optimizer":
            value = str(value).lower()
        setattr(args, arg_key, value)


def build_profiled_args(
    args: argparse.Namespace,
    configs: dict,
    dataset: str,
    site: str | None = None,
) -> argparse.Namespace:
    profiled_args = argparse.Namespace(**vars(args))
    apply_training_profile_values(profiled_args, configs.get("common_training"))
    dataset_profile = configs.get(dataset, {}).get("training_profile", {})
    if dataset == "openneuro":
        apply_training_profile_values(profiled_args, dataset_profile)
    elif dataset == "abide":
        apply_training_profile_values(profiled_args, dataset_profile.get("common_site_training_overrides"))
        if site is not None:
            apply_training_profile_values(profiled_args, dataset_profile.get("sites", {}).get(site))
    return profiled_args


def build_loader(split: ArraySplit, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    effective_batch_size = len(split) if batch_size <= 0 else batch_size
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        GraphFeatureDataset.from_split(split),
        batch_size=effective_batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def move_batch_to_device(batch, device: torch.device):
    structural, functional, adjacency, node_prior, metadata, labels = batch
    return (
        structural.to(device),
        functional.to(device),
        adjacency.to(device),
        node_prior.to(device),
        metadata.to(device),
        labels.to(device),
    )


def mask_features(x: torch.Tensor, mask_rate: float, training: bool) -> torch.Tensor:
    """Randomly zero node features during training."""
    if not training or mask_rate <= 0:
        return x
    keep_mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
    return x * keep_mask


def mixup_batch(
    structural: torch.Tensor,
    functional: torch.Tensor,
    adjacency: torch.Tensor,
    node_prior: torch.Tensor,
    metadata: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply fold-internal mixup to graph features and return mixed labels."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    if lam >= 1.0:
        return structural, functional, adjacency, node_prior, metadata, labels, labels, 1.0
    index = torch.randperm(structural.shape[0], device=device)
    metadata_mixed = lam * metadata + (1.0 - lam) * metadata[index] if metadata.shape[1] > 0 else metadata
    return (
        lam * structural + (1.0 - lam) * structural[index],
        lam * functional + (1.0 - lam) * functional[index],
        lam * adjacency + (1.0 - lam) * adjacency[index],
        lam * node_prior + (1.0 - lam) * node_prior[index],
        metadata_mixed,
        labels,
        labels[index],
        lam,
    )


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Class-balanced focal loss with label smoothing."""
    n_classes = logits.shape[1]
    with torch.no_grad():
        smoothed = torch.full_like(logits, label_smoothing / max(n_classes - 1, 1))
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    log_probs = F.log_softmax(logits, dim=1)
    ce = -(smoothed * log_probs).sum(dim=1)
    p_t = torch.exp(-ce)
    focal = ((1.0 - p_t) ** gamma) * ce
    return (focal * weight[targets]).mean()


def fusion_consistency_loss(
    z_fused: torch.Tensor,
    z_structural: torch.Tensor,
    z_functional_coarse: torch.Tensor,
    z_functional_fine: torch.Tensor,
) -> torch.Tensor:
    """Fusion consistency regularizer used by the paper-facing training profile."""
    stable = torch.maximum(z_structural.abs(), z_functional_coarse.abs())
    strength_loss = F.l1_loss(z_fused.abs(), stable)
    detail_target = (z_functional_fine - z_functional_coarse).abs()
    detail_pred = (z_fused - z_functional_coarse).abs()
    detail_loss = F.l1_loss(detail_pred, detail_target)
    anchor = 0.5 * (z_structural + z_functional_coarse)
    residual_loss = F.l1_loss((z_fused - anchor).abs(), (z_functional_fine - anchor).abs())
    return (strength_loss + detail_loss + residual_loss) / 3.0


def granule_compactness_loss(model: CoRGraNet) -> torch.Tensor:
    """Compactness loss for latent information granule assignments."""
    if not model.training or "membership" not in model.last_outputs:
        return next(model.parameters()).new_tensor(0.0)
    z_for_lig = model.last_outputs["z_for_lig"]
    membership = model.last_outputs["membership"]
    prototypes = model.lig_encoder.prototypes
    distances = torch.cdist(z_for_lig, prototypes)
    return (distances * membership).sum(dim=1).mean()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_factor: float = 0.02,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm-up followed by cosine learning-rate decay."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_loader(
    model: CoRGraNet,
    loader: DataLoader,
    device: torch.device,
    loss_fn=None,
) -> tuple[np.ndarray, float | None]:
    model.eval()
    probs_all = []
    loss_total = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            structural, functional, adjacency, node_prior, metadata, labels = move_batch_to_device(batch, device)
            logits = model(structural, functional, adjacency, node_prior, metadata)
            if loss_fn is not None:
                loss_total += float(loss_fn(logits, labels).item()) * int(labels.shape[0])
                n_samples += int(labels.shape[0])
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    loss = loss_total / max(n_samples, 1) if loss_fn is not None else None
    return np.concatenate(probs_all, axis=0), loss


def evaluate_split(
    model: CoRGraNet,
    split: ArraySplit,
    device: torch.device,
    batch_size: int,
    loss_fn: torch.nn.Module | None = None,
) -> dict:
    loader = build_loader(split, batch_size, shuffle=False)
    probs, loss = evaluate_loader(model, loader, device, loss_fn)
    pred = probs.argmax(axis=1)
    metrics = metrics_from_predictions(split.labels, pred, probs[:, 1])
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def train_one_fold(
    model: CoRGraNet,
    fold_data: TrainFoldData,
    args: argparse.Namespace,
    device: torch.device,
    fold_seed: int,
    save_path: Path,
) -> dict:
    model.to(device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = (
        build_warmup_cosine_scheduler(
            optimizer,
            args.epochs,
            args.warmup_epochs,
            min_factor=args.scheduler_min_factor,
        )
        if args.training_profile == "paper" and not args.no_lr_schedule
        else None
    )
    counts = np.bincount(fold_data.train.labels, minlength=2).astype(np.float32)
    class_weight_np = np.ones(2, dtype=np.float32)
    if not args.no_class_weight:
        if args.training_profile == "paper":
            inverse = (1.0 / np.clip(counts, 1.0, None)).astype(np.float32)
            class_weight_np = inverse / inverse.sum() * 2.0
        else:
            class_weight_np = counts.sum() / (2.0 * np.clip(counts, 1.0, None))
    class_weight = torch.from_numpy(class_weight_np.astype(np.float32)).to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=None if args.no_class_weight else class_weight)

    def validation_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if args.training_profile == "paper":
            return focal_loss(
                logits,
                labels,
                class_weight,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
        return ce_loss_fn(logits, labels)

    train_loader = build_loader(fold_data.train, args.batch_size, shuffle=True, seed=fold_seed)

    best_state = state_dict_to_cpu(model)
    best_epoch = 0
    best_score = -np.inf
    best_loss = np.inf
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            structural, functional, adjacency, node_prior, metadata, labels = move_batch_to_device(batch, device)
            if args.training_profile == "paper":
                structural = mask_features(structural, args.feat_mask_rate, True)
                functional = mask_features(functional, args.feat_mask_rate, True)
                structural, functional, adjacency, node_prior, metadata, labels_a, labels_b, lam = mixup_batch(
                    structural,
                    functional,
                    adjacency,
                    node_prior,
                    metadata,
                    labels,
                    args.mixup_alpha,
                    device,
                )
                logits = model(structural, functional, adjacency, node_prior, metadata)
                loss = lam * focal_loss(
                    logits,
                    labels_a,
                    class_weight,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                ) + (1.0 - lam) * focal_loss(
                    logits,
                    labels_b,
                    class_weight,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )
                if args.fusion_weight > 0:
                    outputs = model.last_outputs
                    loss = loss + args.fusion_weight * fusion_consistency_loss(
                        outputs["z_fused"],
                        outputs["z_structural"],
                        outputs["z_functional_coarse"],
                        outputs["z_functional_fine"],
                    )
                if args.granule_weight > 0:
                    loss = loss + args.granule_weight * granule_compactness_loss(model)
            else:
                logits = model(structural, functional, adjacency, node_prior, metadata)
                loss = ce_loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_total += float(loss.item()) * int(labels.shape[0])

        if scheduler is not None:
            scheduler.step()

        train_loss = loss_total / max(len(fold_data.train), 1)
        validation_metrics = None
        if fold_data.validation is not None:
            validation_metrics = evaluate_split(model, fold_data.validation, device, args.batch_size, validation_loss_fn)
            score = float(validation_metrics["balanced_accuracy"])
            val_loss = float(validation_metrics["loss"])
            improved = score > best_score + args.min_delta or (
                abs(score - best_score) <= args.min_delta and val_loss < best_loss
            )
            if improved:
                best_score = score
                best_loss = val_loss
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_epoch = epoch
            best_state = state_dict_to_cpu(model)

        should_log = args.log_interval > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0)
        if should_log:
            message = f"    epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f}"
            if validation_metrics is not None:
                message += (
                    f" val_ba={validation_metrics['balanced_accuracy']:.4f}"
                    f" val_loss={validation_metrics['loss']:.4f}"
                )
            print(message)

        patience = 30 if args.patience is None else args.patience
        if fold_data.validation is not None and patience > 0 and stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    test_metrics = evaluate_split(model, fold_data.test, device, args.batch_size)
    row = {
        "best_epoch": int(best_epoch),
        "checkpoint": str(save_path),
        "n_train": len(fold_data.train),
        "n_validation": 0 if fold_data.validation is None else len(fold_data.validation),
        "n_test": len(fold_data.test),
    }
    if fold_data.validation is not None:
        row["best_validation_balanced_accuracy"] = float(best_score)
    row.update(test_metrics)
    return row


def evaluate_fold(model: CoRGraNet, checkpoint: Path, fold_data, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    loader = DataLoader(GraphFeatureDataset.from_fold_data(fold_data), batch_size=len(fold_data.labels_test), shuffle=False)
    probs, _ = evaluate_loader(model, loader, device)
    pred = probs.argmax(axis=1)
    return metrics_from_predictions(fold_data.labels_test, pred, probs[:, 1])


def evaluate_openneuro(args: argparse.Namespace, configs: dict, device: torch.device) -> dict:
    cfg = configs["openneuro"]
    data = np.load(args.openneuro_npz, allow_pickle=True)
    labels = data["labels"].astype(np.int64)
    splitter = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    fold_rows = []
    for fold, (_, test_idx) in enumerate(splitter.split(data["mri_node_features"], labels)):
        fold_data = openneuro_fold_from_npz(args.openneuro_npz, args.openneuro_participants_tsv, test_idx)
        model = CoRGraNet(model_config_for_fold(cfg["model"], fold_data))
        checkpoint = checkpoint_path(args.checkpoint_root, cfg, fold=fold)
        row = {"fold": fold}
        row.update(evaluate_fold(model, checkpoint, fold_data, device))
        fold_rows.append(row)
    return {"folds": fold_rows, "summary": summarize_folds(fold_rows)}


def evaluate_abide(args: argparse.Namespace, configs: dict, device: torch.device) -> dict:
    cfg = configs["abide"]
    site_rows = []
    fold_rows = []
    for site in cfg["sites"]:
        dataset = load_abide_cached_site(args.abide_cache_dir, site)
        labels = dataset.labels.astype(np.int64)
        n_splits = min(cfg["n_splits"], int(np.bincount(labels, minlength=2).min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
        site_fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            fold_data = build_abide_fold(dataset, train_idx, test_idx)
            model = CoRGraNet(model_config_for_fold(cfg["model"], fold_data))
            checkpoint = checkpoint_path(args.checkpoint_root, cfg, site=site, fold=fold)
            row = {"site": site, "fold": fold, "n": int(len(labels))}
            row.update(evaluate_fold(model, checkpoint, fold_data, device))
            fold_rows.append(row)
            site_fold_rows.append(row)
        site_rows.append({"site": site, "n": int(len(labels)), "folds": len(site_fold_rows), **summarize_folds(site_fold_rows)})

    weighted = {}
    for metric in ["balanced_accuracy", "mcc", "auc"]:
        mean, std = weighted_site_summary(site_rows, metric)
        weighted[f"{metric}_weighted_mean"] = mean
        weighted[f"{metric}_weighted_std"] = std
    return {"folds": fold_rows, "sites": site_rows, "weighted_summary": weighted}


def run_test(args: argparse.Namespace) -> None:
    configs = load_configs(args.configs)
    device = torch.device(args.device)
    results = {}
    if not args.skip_openneuro:
        results["openneuro"] = evaluate_openneuro(args, configs, device)
    if not args.skip_abide:
        results["abide"] = evaluate_abide(args, configs, device)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def train_openneuro(args: argparse.Namespace, configs: dict, device: torch.device) -> dict:
    cfg = configs["openneuro"]
    train_args = build_profiled_args(args, configs, "openneuro")
    seed = train_args.seed if train_args.seed is not None else cfg["seed"]
    dataset = load_openneuro_npz(args.openneuro_npz, args.openneuro_participants_tsv)
    labels = dataset.labels.astype(np.int64)
    splitter = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    selected_folds = parse_fold_filter(args.fold)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        if selected_folds is not None and fold not in selected_folds:
            continue
        fold_seed = seed + fold
        set_seed(fold_seed)
        fold_data = build_openneuro_training_fold(dataset, train_idx, test_idx, train_args.val_size, fold_seed)
        model = CoRGraNet(model_config_for_split(cfg["model"], fold_data.train))
        checkpoint = training_checkpoint_path(args.output_dir, "openneuro", cfg, fold=fold)
        print(f"  OpenNeuro fold {fold}: train={len(fold_data.train)} val={0 if fold_data.validation is None else len(fold_data.validation)} test={len(fold_data.test)}")
        row = {"dataset": "openneuro", "fold": fold}
        row.update(train_one_fold(model, fold_data, train_args, device, fold_seed, checkpoint))
        fold_rows.append(row)
        print(f"    test_ba={row['balanced_accuracy']:.4f} test_mcc={row['mcc']:.4f} test_auc={row.get('auc', float('nan')):.4f}")
    if not fold_rows:
        raise ValueError("No OpenNeuro folds were selected.")
    return {"folds": fold_rows, "summary": summarize_folds(fold_rows)}


def train_abide(args: argparse.Namespace, configs: dict, device: torch.device) -> dict:
    cfg = configs["abide"]
    selected_sites = parse_site_filter(args.site, cfg["sites"])
    selected_folds = parse_fold_filter(args.fold)
    site_rows = []
    fold_rows = []
    site_offsets = {site: idx for idx, site in enumerate(cfg["sites"])}
    for site in selected_sites:
        train_args = build_profiled_args(args, configs, "abide", site)
        seed = train_args.seed if train_args.seed is not None else cfg["seed"]
        dataset = load_abide_cached_site(args.abide_cache_dir, site)
        labels = dataset.labels.astype(np.int64)
        n_splits = min(cfg["n_splits"], int(np.bincount(labels, minlength=2).min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
        site_fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            if selected_folds is not None and fold not in selected_folds:
                continue
            fold_seed = seed + site_offsets[site] * 100 + fold
            set_seed(fold_seed)
            fold_data = build_abide_training_fold(dataset, train_idx, test_idx, train_args.val_size, fold_seed)
            model = CoRGraNet(model_config_for_split(cfg["model"], fold_data.train))
            checkpoint = training_checkpoint_path(args.output_dir, "abide", cfg, site=site, fold=fold)
            print(
                f"  ABIDE {site} fold {fold}: "
                f"train={len(fold_data.train)} val={0 if fold_data.validation is None else len(fold_data.validation)} test={len(fold_data.test)}"
            )
            row = {"dataset": "abide", "site": site, "fold": fold, "n": int(len(labels))}
            row.update(train_one_fold(model, fold_data, train_args, device, fold_seed, checkpoint))
            fold_rows.append(row)
            site_fold_rows.append(row)
            print(f"    test_ba={row['balanced_accuracy']:.4f} test_mcc={row['mcc']:.4f} test_auc={row.get('auc', float('nan')):.4f}")
        if site_fold_rows:
            site_rows.append({"site": site, "n": int(len(labels)), "folds": len(site_fold_rows), **summarize_folds(site_fold_rows)})
    if not fold_rows:
        raise ValueError("No ABIDE folds were selected.")
    weighted = {}
    for metric in ["balanced_accuracy", "mcc", "auc"]:
        mean, std = weighted_site_summary(site_rows, metric)
        weighted[f"{metric}_weighted_mean"] = mean
        weighted[f"{metric}_weighted_std"] = std
    return {"folds": fold_rows, "sites": site_rows, "weighted_summary": weighted}


def run_train(args: argparse.Namespace) -> None:
    configs = load_configs(args.configs)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "openneuro":
        results = {"openneuro": train_openneuro(args, configs, device)}
    elif args.dataset == "abide":
        results = {"abide": train_abide(args, configs, device)}
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    metrics_path = args.metrics_json or (args.output_dir / f"{args.dataset}_training_results.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Training metrics saved to {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--configs", type=Path, default=PACKAGE_ROOT / "configs.json")
    common.add_argument("--checkpoint-root", type=Path, default=PACKAGE_ROOT)
    common.add_argument("--device", type=str, default="cpu")

    test = subparsers.add_parser("test", parents=[common], help="Evaluate released CoR-GraNet checkpoints.")
    test.add_argument(
        "--openneuro-npz",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs_meg12_cleaned_full_nodeprior/roi_graph_dataset.npz",
    )
    test.add_argument("--openneuro-participants-tsv", type=Path, default=Path("/Volumes/lilei/data/participants.tsv"))
    test.add_argument("--abide-cache-dir", type=Path, default=PROJECT_ROOT / "data_feature_cache/difumo128_global_site_feature_cache")
    test.add_argument("--output-json", type=Path, default=PACKAGE_ROOT / "outputs/table1_reproduction_results.json")
    test.add_argument("--skip-openneuro", action="store_true")
    test.add_argument("--skip-abide", action="store_true")
    test.set_defaults(func=run_test)

    train = subparsers.add_parser("train", parents=[common], help="Train CoR-GraNet on prepared graph features.")
    train.add_argument("--dataset", choices=["openneuro", "abide"], default="openneuro")
    train.add_argument(
        "--openneuro-npz",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs_meg12_cleaned_full_nodeprior/roi_graph_dataset.npz",
    )
    train.add_argument("--openneuro-participants-tsv", type=Path, default=Path("/Volumes/lilei/data/participants.tsv"))
    train.add_argument("--abide-cache-dir", type=Path, default=PROJECT_ROOT / "data_feature_cache/difumo128_global_site_feature_cache")
    train.add_argument("--site", type=str, default="all", help="ABIDE site name, comma-separated sites, or all.")
    train.add_argument("--fold", type=str, default="all", help="Fold index, comma-separated folds, or all.")
    train.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "checkpoints_")
    train.add_argument("--metrics-json", type=Path, default=PACKAGE_ROOT / "checkpoints_/training_results.json")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--training-profile", choices=["paper", "simple"], default="paper")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=8, help="Use 0 or a negative value for full-batch training.")
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    train.add_argument("--val-size", type=float, default=0.2)
    train.add_argument("--patience", type=int, default=None, help="Defaults to 30 for OpenNeuro and 20 for ABIDE; set 0 to disable early stopping.")
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--log-interval", type=int, default=10)
    train.add_argument("--no-class-weight", action="store_true")
    train.add_argument("--warmup-epochs", type=int, default=10)
    train.add_argument("--scheduler-min-factor", type=float, default=0.02)
    train.add_argument("--no-lr-schedule", action="store_true")
    train.add_argument("--mixup-alpha", type=float, default=0.2)
    train.add_argument("--feat-mask-rate", type=float, default=0.1)
    train.add_argument("--label-smoothing", type=float, default=0.05)
    train.add_argument("--focal-gamma", type=float, default=2.0)
    train.add_argument("--fusion-weight", type=float, default=0.2)
    train.add_argument("--granule-weight", type=float, default=0.05)
    train.set_defaults(func=run_train)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["test"])
    if hasattr(args, "output_json"):
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
