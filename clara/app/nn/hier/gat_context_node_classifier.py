import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import BatchNorm, LayerNorm


class GATContextNodeClassifier(nn.Module):
    """
    Node-level classifier with optional context fusion, built on GATv2 (attention) layers.

    Fusion modes:
      - 'none'  : no context; classify using only node embeddings from the GAT backbone.
      - 'concat': concatenate a projected context vector to node embeddings before the head.
      - 'film'  : FiLM modulation (context produces γ and β to modulate node embeddings).

    Backbone:
      [GATv2 -> Norm -> ReLU -> (Dropout)] x L  →  (optional fusion)  →  Head → logits

    Notes:
    - When fusion_mode='none', context_dim can be 0 and ctx_nodes is ignored.
    - When fusion_mode in {'concat','film'}, ctx_nodes must be [N, context_dim].
    - The trainer is responsible for construir/normalizar ctx_nodes (LayerNorm/z-score/L2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        norm_type: str,
        *,
        num_heads: int = 4,  # multi-head only on the first GAT layer
        num_classes: int = 1,
        fusion_mode: str = "concat",  # 'none' | 'concat' | 'film'
        context_dim: int = 0,  # required (>0) for 'concat'/'film'
        use_layer_dropout: bool = True,
        ctx_bottleneck_dim: Optional[
            int
        ] = None,  # e.g., 32 or 64; if None -> min(64, context_dim)
        context_dropout_p: float = 0.10,
        context_scale_init: float = 1.0,
        film_bias: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ("batch", "layer"), "norm_type must be 'batch' or 'layer'"
        fusion_mode = fusion_mode.lower()
        assert fusion_mode in (
            "none",
            "concat",
            "film",
        ), "fusion_mode must be 'none', 'concat' or 'film'"

        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.fusion_mode = fusion_mode
        self.context_dim = int(context_dim)
        self.use_layer_dropout = bool(use_layer_dropout)
        self.num_classes = int(num_classes)
        self.context_dropout_p = float(context_dropout_p)
        self.film_bias = bool(film_bias)
        self.num_heads = int(num_heads)

        # Does this configuration require context?
        self.requires_context = fusion_mode in {"concat", "film"}
        if self.requires_context:
            assert (
                self.context_dim > 0
            ), "context_dim must be > 0 when using 'concat' or 'film'."

        # Learnable global scale for context (useful for annealing/scheduling in the trainer)
        self.context_scale = nn.Parameter(
            torch.tensor(float(context_scale_init)), requires_grad=False
        )

        # --- GATv2 backbone (multi-head only at first layer) ---
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                in_dim = input_dim
                out_dim = hidden_dim
                heads = self.num_heads
                norm_dim = out_dim * heads  # concat on heads
            else:
                in_dim = hidden_dim * self.num_heads if layer_idx == 1 else hidden_dim
                out_dim = hidden_dim
                heads = 1
                norm_dim = out_dim

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=self.dropout,
                    concat=True,  # keep concatenated heads
                )
            )
            norm = BatchNorm(norm_dim) if norm_type == "batch" else LayerNorm(norm_dim)
            self.norm_layers.append(norm)

        # Node embedding width after backbone
        node_emb_dim = (
            hidden_dim if self.num_layers > 1 else hidden_dim * self.num_heads
        )
        self.node_emb_dim = node_emb_dim

        # --- Heads / fusion-specific layers ---
        # Base head (used for 'none' and after FiLM)
        self.base_head = nn.Sequential(
            nn.Linear(node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if self.requires_context:
            # Context projection (shared by both fusion modes) → map to node_emb_dim
            if ctx_bottleneck_dim is None:
                ctx_bottleneck_dim = min(64, self.context_dim)
            assert ctx_bottleneck_dim > 0, "ctx_bottleneck_dim must be > 0"

            self.ctx_proj = nn.Sequential(
                nn.Linear(self.context_dim, ctx_bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(ctx_bottleneck_dim, node_emb_dim),
            )

            if self.fusion_mode == "concat":
                fused_dim = node_emb_dim + node_emb_dim  # [h || ctx_proj]
                self.concat_head = nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
                self.film = None
            else:  # 'film'
                out_dim = 2 * node_emb_dim if self.film_bias else node_emb_dim
                self.film = nn.Linear(node_emb_dim, out_dim)
                self.concat_head = None
        else:
            self.ctx_proj = None
            self.concat_head = None
            self.film = None

    # ---------------------------
    # Core blocks
    # ---------------------------

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute node embeddings after the GAT stack.

        Returns:
            h : [N, node_emb_dim]
        """
        h = x
        for conv, norm in zip(self.gat_layers, self.norm_layers):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if self.use_layer_dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    # ---------------------------
    # Fusion paths
    # ---------------------------

    def fuse_concat(self, h: torch.Tensor, ctx_nodes: torch.Tensor) -> torch.Tensor:
        """
        Concat fusion:
          - project context to node_emb_dim
          - dropout on projected context (regularization)
          - concat with h and run fusion head
        """
        ctx_proj = self.ctx_proj(ctx_nodes)  # [N, node_emb_dim]
        ctx_proj = F.dropout(ctx_proj, p=self.context_dropout_p, training=self.training)
        fused = torch.cat([h, ctx_proj], dim=1)  # [N, 2*node_emb_dim]
        return self.concat_head(fused)

    def fuse_film(self, h: torch.Tensor, ctx_nodes: torch.Tensor) -> torch.Tensor:
        """
        FiLM fusion:
          - project context to node_emb_dim
          - (optional) dropout on projected context
          - generate γ (and β) via a linear layer from context
          - modulate h: h_tilde = h * sigmoid(γ) + β  (β optional)
        """
        ctx_proj = self.ctx_proj(ctx_nodes)  # [N, node_emb_dim]
        ctx_proj = F.dropout(ctx_proj, p=self.context_dropout_p, training=self.training)

        if self.film_bias:
            gb = self.film(ctx_proj)  # [N, 2*node_emb_dim]
            gamma, beta = gb.chunk(2, dim=1)  # [N,D], [N,D]
        else:
            gamma = self.film(ctx_proj)  # [N,D]
            beta = torch.zeros_like(gamma)  # no bias term

        # global scale for stability/annealing
        gamma = gamma * self.context_scale
        beta = beta * self.context_scale

        h_tilde = h * torch.sigmoid(gamma) + beta  # FiLM modulation
        return self.base_head(h_tilde)

    # ---------------------------
    # Forward
    # ---------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        ctx_nodes: Optional[
            torch.Tensor
        ] = None,  # required only if fusion_mode!='none'
    ) -> torch.Tensor:
        """
        Compute per-node logits.

        Args:
            x         : [N, input_dim] node features.
            edge_index: [2, E] graph connectivity (COO).
            ctx_nodes : [N, context_dim] per-node context; required for 'concat'/'film',
                        ignored for 'none'.

        Returns:
            logits : [N, num_classes] raw logits (no activation).
        """
        h = self.get_node_embeddings(x, edge_index)  # [N, node_emb_dim]

        if self.fusion_mode == "none":
            return self.base_head(h)

        # Context is required for 'concat'/'film'
        assert (
            ctx_nodes is not None
        ), "ctx_nodes must be provided for 'concat'/'film' fusion."
        assert (
            ctx_nodes.dim() == 2
            and ctx_nodes.size(0) == h.size(0)
            and ctx_nodes.size(1) == self.context_dim
        ), f"ctx_nodes must be [N, {self.context_dim}]"

        if self.fusion_mode == "concat":
            return self.fuse_concat(h, ctx_nodes)
        else:  # 'film'
            return self.fuse_film(h, ctx_nodes)

    # ---------------------------
    # Utilities
    # ---------------------------

    @torch.no_grad()
    def set_context_scale(self, value: float):
        """
        Set (without tracking gradients) the global context scale used in fusion.
        Useful for simple annealing schedules in the trainer.
        """
        self.context_scale.fill_(float(value))
