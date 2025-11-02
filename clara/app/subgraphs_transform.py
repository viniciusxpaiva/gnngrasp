"""
subgraph_transforms.py — Minimal pluggable interface for subgraph-level transforms
(E1 variant: structural node features only)

This module defines a tiny, extensible interface to attach additional views and
subgraph descriptors to PyG `Data` objects, *after* you generate ego-subgraphs.

Current contents:
- Abstract base: `SubgraphTransform`
- Orchestrator: `apply_subgraph_transforms`
- Normalization helpers: `FeatureNormalizer`
- E1 implementation: `NodeStructTransform` (degree, log-degree, clustering, PR, core proxy)

Design notes:
- **Non-invasive**: We do not overwrite `data.x`. New features go to `data.x_view_*`.
- **Composable**: Multiple transforms can be chained; each one returns a dict of new fields.
- **Cache-ready**: Transforms are pure functions of `Data`. You can persist results by hashing
  subgraph ids + config and saving `Data` via `torch.save`.
- **Scalable**: Uses pure PyTorch; clustering uses a dense adjacency only when the subgraph is
  small enough (configurable threshold). For larger subgraphs, clustering is skipped or set to 0.
- **Global vs Local**: You can toggle `use_global=True` if you want to compute structural features
  on the global graph and then slice per-subgraph (requires passing the global graph + a mapping).
  The default here is local computation on the subgraph itself.

Example usage (after building train/test subgraphs):

    transforms = [
        NodeStructTransform(
            use_global=False,
            max_nodes_for_dense=2000,
            normalize={"method": "layernorm"}
        ),
    ]
    train_subgraphs = apply_subgraph_transforms(train_subgraphs_raw, transforms)
    test_subgraphs  = apply_subgraph_transforms(test_subgraphs_raw, transforms)

Then, in your GNN1 pipeline you can select `x_key = "x_view_struct"` to use these features;
GNN2 remains unchanged and may inject a `z_desc` when you add other transforms (E2+).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# -----------------------------------------------------------------------------
# Base interface
# -----------------------------------------------------------------------------
class SubgraphTransform:
    """Abstract callable that augments a PyG `Data` with extra fields.

    Contract: `__call__(data) -> Dict[str, Tensor]`
    - Must NOT mutate `data` in-place; return a dict with new tensors instead.
    - The orchestrator will attach returned fields to `data`.
    - Keep tensors on CPU by default (loader/collate can move to device later).
    """

    def __call__(self, data: Data) -> Dict[str, Tensor]:  # pragma: no cover
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Normalization helpers (per-subgraph)
# -----------------------------------------------------------------------------
@dataclass
class NormalizationConfig:
    method: str = "none"  # "none" | "standard" | "layernorm"
    eps: float = 1e-6


class FeatureNormalizer:
    """Applies simple feature normalization per subgraph.

    Options:
    - "none": return as-is
    - "standard": z-score per feature (mean=0, std=1) computed over nodes
    - "layernorm": LayerNorm-style (mean/std over feature dim) for each node row
    """

    def __init__(self, cfg: Optional[NormalizationConfig] = None) -> None:
        self.cfg = cfg or NormalizationConfig()

    def __call__(self, x: Tensor) -> Tensor:
        method = self.cfg.method.lower()
        eps = self.cfg.eps
        if method == "none":
            return x
        if method == "standard":
            # Compute stats over the node dimension (N x D -> stats over N)
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, unbiased=False, keepdim=True)
            return (x - mean) / (std + eps)
        if method == "layernorm":
            # Normalize each row independently (N x D -> stats over D)
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, unbiased=False, keepdim=True)
            return (x - mean) / (std + eps)
        raise ValueError(f"Unknown normalization method: {self.cfg.method}")


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def apply_subgraph_transforms(
    subgraphs: Iterable[Data],
    transforms: Iterable[SubgraphTransform],
    *,
    compose_descriptor: bool = True,
) -> List[Data]:
    """Apply a sequence of transforms to each subgraph and attach new fields.

    Parameters
    ----------
    subgraphs : Iterable[Data]
        Ego-subgraphs already constructed (e.g., via k-hop or coloring policy).
    transforms : Iterable[SubgraphTransform]
        Instances to run; each returns a dict of {field_name: tensor}.
    compose_descriptor : bool, default=True
        If future transforms expose `z_desc_*` 1-D tensors, this flag will
        concatenate them into a single `data.z_desc`. (No-op in E1.)

    Returns
    -------
    List[Data]
        New list with the same Data objects, now bearing extra attributes.
    """
    out: List[Data] = []
    for d in subgraphs:
        extra: Dict[str, Tensor] = {}
        for t in transforms:
            fields = t(d)
            if not isinstance(fields, dict):
                raise TypeError(
                    f"Transform {t.__class__.__name__} must return a dict, got {type(fields)}"
                )
            extra.update(fields)
        # Attach fields
        for k, v in extra.items():
            setattr(d, k, v)
        # Optional descriptor composition (future-proof)
        if compose_descriptor:
            z_parts: List[Tensor] = []
            for k, v in extra.items():
                if k.startswith("z_desc_"):
                    v1 = v.view(-1) if v.dim() > 1 else v
                    z_parts.append(v1)
            if len(z_parts) > 0:
                d.z_desc = torch.cat(z_parts, dim=0)
        out.append(d)
    return out


# -----------------------------------------------------------------------------
# E1 — Structural node features (local subgraph computation)
# -----------------------------------------------------------------------------
@dataclass
class NodeStructConfig:
    """Configuration for E1 structural node features.

    Parameters
    ----------
    use_global : bool
        If True, features are assumed to be pre-computed on the *global* graph
        and then sliced for the subgraph. This file implements only the *local*
        computation. To support global, extend with a `GlobalProvider`.
    max_nodes_for_dense : int
        Maximum subgraph size to build a dense adjacency for clustering. Above
        this threshold, clustering is skipped (zeros).
    normalize : Optional[Dict]
        Normalization config passed to `FeatureNormalizer`.
    include : Optional[List[str]]
        Which features to include; default selects all implemented.
    """

    use_global: bool = False
    max_nodes_for_dense: int = 2000
    normalize: Optional[Dict] = None
    include: Optional[List[str]] = (
        None  # e.g., ["deg", "logdeg", "cluster", "pr", "core"]
    )


class NodeStructTransform(SubgraphTransform):
    """Compute structural node features for E1 and expose them as `x_view_struct`.

    Features (per node):
    - deg     : degree (in undirected view)
    - logdeg  : log(1 + degree)
    - cluster : local clustering coefficient (approx; requires dense A)
    - pr      : PageRank (power iteration, 10 iters)
    - core    : cheap proxy for core number (degree quantile bin)

    Notes:
    - Everything is computed *locally on the subgraph* by default.
    - Clustering is set to 0 if the subgraph is larger than `max_nodes_for_dense`.
    - Tensors are kept on CPU; the training loop should move them to the device.
    - Additional signals (e.g., harmonic closeness, betweenness) can be added later.
    """

    FEAT_ORDER = ("deg", "logdeg", "cluster", "pr", "core")

    def __init__(self, *, cfg: Optional[NodeStructConfig] = None, **kwargs) -> None:
        if cfg is None:
            # Allow legacy: pass kwargs directly to config fields
            cfg = NodeStructConfig(**kwargs)
        self.cfg = cfg
        norm_cfg = None
        if cfg.normalize is not None:
            norm_cfg = NormalizationConfig(**cfg.normalize)
        self.normalizer = FeatureNormalizer(norm_cfg)
        # Pre-select indices for included features
        if cfg.include is None:
            self._keep_idx = list(range(len(self.FEAT_ORDER)))
        else:
            name_to_idx = {n: i for i, n in enumerate(self.FEAT_ORDER)}
            self._keep_idx = [name_to_idx[n] for n in cfg.include if n in name_to_idx]
            if len(self._keep_idx) == 0:
                raise ValueError(
                    "`include` filtered out all features; nothing to keep."
                )

    def __call__(self, data: Data) -> Dict[str, Tensor]:
        if self.cfg.use_global:
            raise NotImplementedError(
                "Global-mode is not implemented in this file. Provide a GlobalProvider to slice features."
            )
        x_struct = self._compute_local_struct_features(
            edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        # Select columns if needed
        x_struct = x_struct[:, self._keep_idx]
        # Normalize per subgraph if requested
        x_struct = self.normalizer(x_struct)
        return {"x_view_struct": x_struct}

    # ---------------------------- internals ----------------------------
    def _compute_local_struct_features(
        self, edge_index: Tensor, num_nodes: int
    ) -> Tensor:
        # Ensure undirected, remove (optional) self-loops via PyG utilities if needed.
        ei = to_undirected(edge_index)
        N = int(num_nodes)
        device = ei.device

        # Degree (COO bincount over source indices)
        deg = torch.bincount(ei[0], minlength=N).to(torch.float32)
        logdeg = torch.log1p(deg)

        # Local clustering coefficient via dense adjacency if feasible
        if N <= self.cfg.max_nodes_for_dense:
            A = torch.zeros((N, N), device=device)
            A[ei[0], ei[1]] = 1
            # Symmetrize (already undirected, but ensure)
            A = torch.maximum(A, A.T)
            # Count triangles per node ~ diag(A^3)/2 (exact for simple graphs)
            # Use (A@A)*A pattern to reduce a GEMM
            A2 = A @ A
            tri = (A2 * A).sum(dim=1) / 2.0
            denom = torch.clamp(deg * (deg - 1.0), min=1.0)
            cluster = (2.0 * tri) / denom
            cluster = cluster.to(torch.float32)
        else:
            cluster = torch.zeros(N, dtype=torch.float32, device=device)

        # PageRank (10 power iterations). Use column-stochastic formulation.
        pr = torch.full((N,), 1.0 / max(N, 1), device=device, dtype=torch.float32)
        alpha = 0.85
        # Precompute out-degree to avoid divides by zero
        out_deg = torch.clamp(deg, min=1.0)
        for _ in range(10):
            # pr_new[v] += pr[u]/deg[u] for each edge u->v (we have undirected edges both ways)
            pr_new = torch.zeros_like(pr)
            pr_new.index_add_(0, ei[1], pr[ei[0]] / out_deg[ei[0]])
            pr = (1 - alpha) / max(N, 1) + alpha * pr_new

        # Core number proxy via degree quantiles (3 bins -> {0,1,2,3})
        # If N < 4, fall back to zeros to avoid degenerate quantiles.
        if N >= 4:
            q = torch.quantile(deg, torch.tensor([0.25, 0.5, 0.75], device=device))
            core = (
                (deg >= q[2]).float() * 3.0
                + ((deg >= q[1]) & (deg < q[2])).float() * 2.0
                + ((deg >= q[0]) & (deg < q[1])).float() * 1.0
            )
        else:
            core = torch.zeros(N, dtype=torch.float32, device=device)

        # Stack in canonical order (deg, logdeg, cluster, pr, core) and move to CPU
        x = torch.stack([deg, logdeg, cluster, pr, core], dim=1).to("cpu")
        return x


# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
