import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm


class GCNBiasNodeClassifier(nn.Module):
    """
    Node-level classifier with optional bias fusion, built on GCN layers.

    Supported bias strategies:
      1) 'concat'  : concatenate a per-node context vector to node embeddings
                     before the classifier (e.g., subgraph embedding expanded to nodes).
                     This version includes a learnable context projection + context dropout
                     to stabilize training and avoid context dominance.

    Base architecture:
      [GCN -> Norm -> ReLU -> (Dropout)] x L  →  (fusion)  →  MLP → logits

    Notes
    -----
    - If bias_mode='none', behaves like a standard node classifier.
    - If bias_mode='concat', pass `ctx_nodes` to forward() with shape [N, context_dim].
      The class will internally project `ctx_nodes` to match `hidden_dim` and apply
      a small dropout on the projected context before the final head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        norm_type: str,
        num_classes: int = 1,
        bias_mode: str = "none",  # 'none' | 'concat'
        context_dim: int = 0,  # required if bias_mode == 'concat'
        use_layer_dropout: bool = True,  # apply dropout between GCN layers
        # ---- Improvements for 'concat' mode ----
        ctx_bottleneck_dim: (
            int | None
        ) = None,  # e.g., 32 or 64; if None -> min(64, context_dim)
        context_dropout_p: float = 0.10,  # dropout applied AFTER context projection
    ):
        """
        Args:
            input_dim      : Node feature dimension.
            hidden_dim     : Hidden dimension for GCN stack and classifier head.
            dropout        : Dropout prob used in MLPs and (optionally) between GCN layers.
            num_layers     : Number of GCN layers (>=1).
            norm_type      : 'batch' or 'layer' normalization after each GCN layer.
            num_classes    : Output dimension (1 for binary, C for multi-class).
            bias_mode      : 'none' | 'concat'.
            context_dim    : Dimensionality of per-node context (only for 'concat').
            use_layer_dropout : If True, applies dropout after each layer activation.
            ctx_bottleneck_dim : Bottleneck dimension for context projection before
                                 matching hidden_dim. If None, defaults to min(64, context_dim).
            context_dropout_p  : Dropout applied to the projected context vector
                                 before fusion (helps regularize the context).
        """
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ("batch", "layer"), "norm_type must be 'batch' or 'layer'"
        assert bias_mode in (
            "none",
            "concat",
        ), "bias_mode must be 'none' or 'concat'"
        if bias_mode == "concat":
            assert context_dim > 0, "context_dim must be > 0 when bias_mode='concat'"

        self.dropout = dropout
        self.num_layers = num_layers
        self.bias_mode = bias_mode
        self.context_dim = context_dim
        self.use_layer_dropout = use_layer_dropout
        self.num_classes = num_classes

        # --- GCN backbone ---
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            out_dim = hidden_dim
            self.gcn_layers.append(GCNConv(in_channels=in_dim, out_channels=out_dim))
            norm = BatchNorm(out_dim) if norm_type == "batch" else LayerNorm(out_dim)
            self.norm_layers.append(norm)
            in_dim = out_dim  # next layer input

        # Final node embedding dimension after GCN stack
        node_emb_dim = hidden_dim

        # --- Classifier head (no fusion) ---
        self.base_head = nn.Sequential(
            nn.Linear(node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),  # logits
        )

        # --- Concat pathway: context projection + fusion head ---
        if self.bias_mode == "concat":
            # Choose a small bottleneck to avoid letting the raw context dominate.
            if ctx_bottleneck_dim is None:
                ctx_bottleneck_dim = min(64, context_dim)
            assert (
                ctx_bottleneck_dim > 0
            ), "ctx_bottleneck_dim must be > 0 for 'concat' mode"

            self.context_dropout_p = float(context_dropout_p)
            self.ctx_proj = nn.Sequential(
                nn.Linear(context_dim, ctx_bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ctx_bottleneck_dim, hidden_dim),  # map context -> hidden_dim
            )

            # After projection, we fuse [h, ctx_proj] with a compact head
            fused_dim = node_emb_dim + hidden_dim  # [h || ctx_proj(h)]
            self.concat_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.context_dropout_p = 0.0
            self.ctx_proj = None
            self.concat_head = None

    # ---------------------------------------------------------
    # Core building blocks
    # ---------------------------------------------------------

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute node embeddings after the GCN stack.

        Args:
            x          : [N, input_dim] node features.
            edge_index : [2, E] graph connectivity (COO).

        Returns:
            h : [N, hidden_dim] node embeddings.
        """
        h = x
        for conv, norm in zip(self.gcn_layers, self.norm_layers):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if self.use_layer_dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def node_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply the base classifier head to node embeddings (no context fusion).

        Args:
            h : [N, hidden_dim]

        Returns:
            logits : [N, num_classes]
        """
        return self.base_head(h)

    def node_head_concat(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Apply the concat head to fused node + (projected) context embeddings.

        Args:
            h   : [N, hidden_dim]
            ctx : [N, context_dim] (raw context; will be projected internally)

        Returns:
            logits : [N, num_classes]
        """
        assert (
            self.concat_head is not None and self.ctx_proj is not None
        ), "Concat path not initialized; set bias_mode='concat'."

        # Project context to hidden_dim, then regularize it a bit
        ctx_proj = self.ctx_proj(ctx)  # [N, hidden_dim]
        ctx_proj = F.dropout(ctx_proj, p=self.context_dropout_p, training=self.training)

        fused = torch.cat([h, ctx_proj], dim=1)  # [N, hidden_dim + hidden_dim]
        return self.concat_head(fused)

    # ---------------------------------------------------------
    # Forward with optional bias fusion
    # ---------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ctx_nodes: (
            torch.Tensor | None
        ) = None,  # used if bias_mode='concat'  (shape [N, context_dim])
    ) -> torch.Tensor:
        """
        Compute per-node logits with optional bias fusion.

        Args:
            x            : [N, input_dim] node features.
            edge_index   : [2, E] graph connectivity (COO).
            ctx_nodes    : Optional context per node  (N x context_dim). Only for bias_mode='concat'.

        Returns:
            logits : [N, num_classes] raw logits (no activation).
        """
        h = self.get_node_embeddings(x, edge_index)  # [N, hidden_dim]

        if self.bias_mode == "concat":
            # Expect normalized context to be provided by the trainer if desired
            assert (
                ctx_nodes is not None
            ), "ctx_nodes must be provided when bias_mode='concat'."
            assert (
                ctx_nodes.dim() == 2
                and ctx_nodes.size(0) == h.size(0)
                and ctx_nodes.size(1) == self.context_dim
            ), f"ctx_nodes must be [N, {self.context_dim}]"
            logits = self.node_head_concat(h, ctx_nodes)

        else:
            # Base logits
            logits = self.node_head(h)
        return logits
