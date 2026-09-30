#!/usr/bin/env python3
"""CoR-GraNet model implementation using paper-facing module names."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = torch.abs(adjacency)
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0)
    adjacency = adjacency + eye.expand_as(adjacency)
    degree = adjacency.sum(dim=-1)
    degree_inv_sqrt = degree.clamp_min(1e-6).pow(-0.5)
    return degree_inv_sqrt.unsqueeze(-1) * adjacency * degree_inv_sqrt.unsqueeze(-2)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.bmm(adjacency, x))


class GraphTopologyAttentionEngine(nn.Module):
    """Graph topology attention encoder (GTAE)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        lp_p: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, embed_dim)
        self.topo_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.graph_proj = nn.Linear(embed_dim, embed_dim)
        self.lp_p = lp_p
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        h = self.norm1(h + F.gelu(self.conv1(h, adjacency)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm2(h + F.gelu(self.conv2(h, adjacency)))
        z_nodes = self.node_proj(h)
        attn = torch.softmax(self.topo_attn(z_nodes), dim=1)
        lp_pool = (z_nodes.abs().pow(self.lp_p) * attn).sum(1).pow(1.0 / self.lp_p)
        attn_pool = (attn * z_nodes).sum(1)
        z_graph = self.graph_proj(0.5 * (lp_pool + attn_pool))
        return z_graph, z_nodes


class FunctionalScaleFusion(nn.Module):
    """Gated aggregation of coarse and fine functional graph embeddings."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([coarse, fine], dim=1)
        gate = self.gate(cat)
        return gate * coarse + (1.0 - gate) * fine + self.refine(cat)


class TopologyRefinement(nn.Module):
    """Encodes node-prior features into an additive edge-bias matrix."""

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.graph_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_nodes = self.node_mlp(x)
        z_graph = self.graph_proj(z_nodes.mean(dim=1))
        gate = self.edge_gate(z_nodes).squeeze(-1)
        edge_bias = torch.bmm(gate.unsqueeze(2), gate.unsqueeze(1))
        return z_graph, edge_bias


class LIGEncoder(nn.Module):
    """Soft assignment to K latent information granules (LIGs)."""

    def __init__(self, embed_dim: int, n_granules: int = 3, temperature: float = 1.0) -> None:
        super().__init__()
        self.n_granules = n_granules
        self.prototypes = nn.Parameter(torch.randn(n_granules, embed_dim))
        nn.init.orthogonal_(self.prototypes)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_refined = self.refine(z)
        distances = torch.cdist(z_refined, self.prototypes)
        membership = F.softmax(-distances / self.log_temp.exp(), dim=1)
        granule_id = membership.argmax(dim=1)
        return membership, granule_id, distances


class GranuleAwareFusion(nn.Module):
    """Subject-adaptive structural-functional fusion using LIG memberships."""

    def __init__(self, embed_dim: int, n_granules: int = 3) -> None:
        super().__init__()
        self.n_granules = n_granules
        self.fusion_weights = nn.Parameter(torch.ones(n_granules, 2))
        nn.init.uniform_(self.fusion_weights, 0.3, 0.7)
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        z_structural: torch.Tensor,
        z_functional: torch.Tensor,
        membership: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = membership @ F.softmax(self.fusion_weights, dim=1)
        w_structural = weights[:, 0:1]
        w_functional = weights[:, 1:2]
        z_weighted = w_structural * z_structural + w_functional * z_functional
        z_fused = self.fusion_proj(torch.cat([z_weighted, z_structural * z_functional], dim=1))
        return z_fused, weights


class CrossModalRepresentationReuse(nn.Module):
    """Cross-modal representation reuse used across LIG assignment, fusion, and prediction input construction."""

    def __init__(self) -> None:
        super().__init__()

    def lig_input(self, z_structural: torch.Tensor, z_functional: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_structural, z_functional, metadata], dim=1)

    def fusion_inputs(
        self,
        z_structural: torch.Tensor,
        z_functional: torch.Tensor,
        membership: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return z_structural, z_functional, membership

    def prediction_input(
        self,
        z_fused: torch.Tensor,
        z_structural: torch.Tensor,
        z_functional: torch.Tensor,
        metadata: torch.Tensor,
        membership: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([z_fused, z_structural, z_functional, metadata, membership], dim=1)


class IQWeightedMetadataAdapter(nn.Module):
    """Learnable FIQ/VIQ/PIQ fusion used by the ABIDE Table 1 checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(9, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        age = metadata[:, 0:1]
        sex = metadata[:, 1:2]
        hand = metadata[:, 2:3]
        iq_values = metadata[:, 3:6]
        iq_missing = metadata[:, 6:9]
        logits = self.gate(metadata)
        logits = logits.masked_fill(iq_missing > 0.5, -1e4)
        all_missing = (iq_missing > 0.5).all(dim=1, keepdim=True)
        logits = torch.where(all_missing, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1)
        iq_values = torch.nan_to_num(iq_values, nan=0.0)
        weighted_iq = (weights * iq_values).sum(dim=1, keepdim=True)
        weighted_iq = torch.where(all_missing, torch.zeros_like(weighted_iq), weighted_iq)
        return torch.cat([age, sex, hand, weighted_iq, age * weighted_iq, weighted_iq.pow(2)], dim=1)


class CoRGraNetBackbone(nn.Module):
    """Structural, coarse-functional, and fine-functional GTAE backbone."""

    def __init__(
        self,
        structural_node_dim: int,
        functional_node_dim: int,
        n_coarse: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float,
        node_prior_dim: int,
        edge_prior_weight: float,
        lp_p: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_coarse = n_coarse
        self.edge_prior_weight = edge_prior_weight
        self.structural_encoder = GraphTopologyAttentionEngine(
            structural_node_dim, hidden_dim, embed_dim, lp_p, dropout
        )
        self.functional_coarse_encoder = GraphTopologyAttentionEngine(
            functional_node_dim, hidden_dim, embed_dim, lp_p, dropout
        )
        self.functional_fine_encoder = GraphTopologyAttentionEngine(
            functional_node_dim, hidden_dim, embed_dim, lp_p, dropout
        )
        self.functional_scale_fusion = FunctionalScaleFusion(embed_dim)
        self.topology_refinement = (
            TopologyRefinement(node_prior_dim, hidden_dim, embed_dim)
            if node_prior_dim > 0
            else None
        )

    def forward(
        self,
        structural_nodes: torch.Tensor,
        functional_nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.topology_refinement is not None and self.edge_prior_weight > 0:
            if node_prior is None:
                raise ValueError("node_prior is required when topology_refinement is enabled")
            _, edge_bias = self.topology_refinement(node_prior)
            adjacency = normalize_adjacency(adjacency + self.edge_prior_weight * edge_bias)
        else:
            adjacency = normalize_adjacency(adjacency)

        z_structural, z_structural_nodes = self.structural_encoder(structural_nodes, adjacency)

        coarse_idx = torch.arange(self.n_coarse, device=functional_nodes.device)
        functional_coarse = functional_nodes.index_select(1, coarse_idx)
        adjacency_coarse = adjacency.index_select(1, coarse_idx).index_select(2, coarse_idx)
        z_functional_coarse, _ = self.functional_coarse_encoder(functional_coarse, adjacency_coarse)
        z_functional_fine, z_functional_nodes = self.functional_fine_encoder(functional_nodes, adjacency)
        z_functional = self.functional_scale_fusion(z_functional_coarse, z_functional_fine)

        return {
            "z_structural": z_structural,
            "z_structural_nodes": z_structural_nodes,
            "z_functional": z_functional,
            "z_functional_nodes": z_functional_nodes,
            "z_functional_coarse": z_functional_coarse,
            "z_functional_fine": z_functional_fine,
        }


@dataclass(frozen=True)
class CoRGraNetConfig:
    structural_node_dim: int
    functional_node_dim: int
    n_nodes: int
    n_coarse: int
    hidden_dim: int = 128
    embed_dim: int = 64
    dropout: float = 0.2
    node_prior_dim: int = 0
    edge_prior_weight: float = 0.0
    metadata_dim: int = 6
    raw_metadata_dim: int = 0
    n_granules: int = 3
    granule_temp: float = 1.0
    lp_p: float = 2.0


class CoRGraNet(nn.Module):
    """Chain-of-Representation Granular Network."""

    def __init__(self, config: CoRGraNetConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = CoRGraNetConfig(**config)
        self.config = config
        self.metadata_adapter = IQWeightedMetadataAdapter() if config.raw_metadata_dim == 9 else None
        self.backbone = CoRGraNetBackbone(
            structural_node_dim=config.structural_node_dim,
            functional_node_dim=config.functional_node_dim,
            n_coarse=config.n_coarse,
            hidden_dim=config.hidden_dim,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
            node_prior_dim=config.node_prior_dim,
            edge_prior_weight=config.edge_prior_weight,
            lp_p=config.lp_p,
        )
        lig_input_dim = config.embed_dim * 2 + config.metadata_dim
        self.lig_encoder = LIGEncoder(lig_input_dim, config.n_granules, config.granule_temp)
        self.granule_aware_fusion = GranuleAwareFusion(config.embed_dim, config.n_granules)
        classifier_dim = config.embed_dim * 3 + config.metadata_dim + config.n_granules
        self.cross_modal_representation_reuse = CrossModalRepresentationReuse()
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 2),
        )
        self.last_outputs: dict[str, torch.Tensor] = {}

    def adapt_metadata(self, metadata: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.metadata_adapter is not None:
            if metadata is None:
                metadata = torch.zeros(batch_size, 9, device=device)
            return self.metadata_adapter(metadata)
        if metadata is None:
            return torch.zeros(batch_size, self.config.metadata_dim, device=device)
        return metadata

    def encode(
        self,
        structural_nodes: torch.Tensor,
        functional_nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_prior: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        metadata = self.adapt_metadata(metadata, structural_nodes.shape[0], structural_nodes.device)
        encoded = self.backbone(structural_nodes, functional_nodes, adjacency, node_prior)
        z_structural = encoded["z_structural"]
        z_functional = encoded["z_functional"]
        z_for_lig = self.cross_modal_representation_reuse.lig_input(z_structural, z_functional, metadata)
        membership, granule_id, distances = self.lig_encoder(z_for_lig)
        z_structural_reused, z_functional_reused, membership = self.cross_modal_representation_reuse.fusion_inputs(
            z_structural,
            z_functional,
            membership,
        )
        z_fused, fusion_weights = self.granule_aware_fusion(z_structural_reused, z_functional_reused, membership)
        z_joint = self.cross_modal_representation_reuse.prediction_input(
            z_fused,
            z_structural,
            z_functional,
            metadata,
            membership,
        )
        outputs = {
            **encoded,
            "metadata": metadata,
            "membership": membership,
            "q_lig": membership,
            "granule_id": granule_id,
            "distances": distances,
            "fusion_weights": fusion_weights,
            "z_fused": z_fused,
            "z_joint": z_joint,
            "z_for_lig": z_for_lig,
            "z_structural_reused": z_structural_reused,
            "z_functional_reused": z_functional_reused,
        }
        self.last_outputs = outputs
        return outputs

    def forward(
        self,
        structural_nodes: torch.Tensor,
        functional_nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_prior: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encode(structural_nodes, functional_nodes, adjacency, node_prior, metadata)
        return self.classifier(outputs["z_joint"])


def build_model(config: dict | CoRGraNetConfig) -> CoRGraNet:
    return CoRGraNet(config)
