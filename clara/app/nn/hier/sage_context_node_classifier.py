import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm


class SAGEContextNodeClassifier(nn.Module):
    """
    Node-level classifier with optional context fusion, built on GraphSAGE layers.

    Fusion modes:
      - 'none'  : no context; classifica apenas com embeddings dos nós (backbone).
      - 'concat': concatena um vetor de contexto projetado ao embedding do nó antes do head.
      - 'film'  : FiLM modulation — contexto gera γ (e β opcional) que modulam o embedding.

    Backbone:
      [SAGEConv -> Norm -> ReLU -> (Dropout)] x L  →  (opcional fusão)  →  Head → logits

    Observações:
    - Com fusion_mode='none', context_dim pode ser 0 e `ctx_nodes` é ignorado.
    - Com fusion_mode in {'concat','film'}, `ctx_nodes` deve ser [N, context_dim].
    - A normalização/ablação dos vetores de contexto fica a cargo do trainer (LayerNorm/z-score/L2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        norm_type: str,
        *,
        aggr: str = "mean",  # 'mean' | 'max' | 'add' | 'sum' | 'lstm'
        num_classes: int = 1,
        fusion_mode: str = "concat",  # 'none' | 'concat' | 'film'
        context_dim: int = 0,  # >0 quando usar 'concat'/'film'
        use_layer_dropout: bool = True,
        ctx_bottleneck_dim: Optional[int] = None,  # se None -> min(64, context_dim)
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
        ), "fusion_mode must be 'none', 'concat' ou 'film'"

        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.aggr = aggr
        self.fusion_mode = fusion_mode
        self.context_dim = int(context_dim)
        self.use_layer_dropout = bool(use_layer_dropout)
        self.num_classes = int(num_classes)
        self.context_dropout_p = float(context_dropout_p)
        self.film_bias = bool(film_bias)

        self.requires_context = fusion_mode in {"concat", "film"}
        if self.requires_context:
            assert (
                self.context_dim > 0
            ), "context_dim deve ser > 0 para 'concat'/'film'."

        # Escala global (não-treinável) do contexto — útil para annealing no trainer
        self.context_scale = nn.Parameter(
            torch.tensor(float(context_scale_init)), requires_grad=False
        )

        # --- GraphSAGE backbone ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = input_dim
        for _ in range(self.num_layers):
            self.convs.append(
                SAGEConv(in_channels=in_dim, out_channels=hidden_dim, aggr=self.aggr)
            )
            norm = (
                BatchNorm(hidden_dim) if norm_type == "batch" else LayerNorm(hidden_dim)
            )
            self.norms.append(norm)
            in_dim = hidden_dim

        node_emb_dim = hidden_dim
        self.node_emb_dim = node_emb_dim

        # --- Head base (usado em 'none' e pós-FiLM) ---
        self.base_head = nn.Sequential(
            nn.Linear(node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # --- Partes dependentes de contexto ---
        if self.requires_context:
            if ctx_bottleneck_dim is None:
                ctx_bottleneck_dim = min(64, self.context_dim)
            assert ctx_bottleneck_dim > 0, "ctx_bottleneck_dim deve ser > 0"

            # Projeção do contexto para a mesma largura do embedding do nó
            self.ctx_proj = nn.Sequential(
                nn.Linear(self.context_dim, ctx_bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(ctx_bottleneck_dim, node_emb_dim),
            )

            if self.fusion_mode == "concat":
                fused_dim = node_emb_dim + node_emb_dim
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
    # Core
    # ---------------------------

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        GraphSAGE stack → embeddings por nó [N, node_emb_dim]
        """
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if self.use_layer_dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    # ---------------------------
    # Fusão
    # ---------------------------

    def fuse_concat(self, h: torch.Tensor, ctx_nodes: torch.Tensor) -> torch.Tensor:
        ctx_proj = self.ctx_proj(ctx_nodes)  # [N, D]
        ctx_proj = F.dropout(ctx_proj, p=self.context_dropout_p, training=self.training)
        fused = torch.cat([h, ctx_proj], dim=1)  # [N, 2D]
        return self.concat_head(fused)

    def fuse_film(self, h: torch.Tensor, ctx_nodes: torch.Tensor) -> torch.Tensor:
        ctx_proj = self.ctx_proj(ctx_nodes)  # [N, D]
        ctx_proj = F.dropout(ctx_proj, p=self.context_dropout_p, training=self.training)

        if self.film_bias:
            gb = self.film(ctx_proj)  # [N, 2D]
            gamma, beta = gb.chunk(2, dim=1)  # [N,D], [N,D]
        else:
            gamma = self.film(ctx_proj)  # [N,D]
            beta = torch.zeros_like(gamma)

        gamma = gamma * self.context_scale
        beta = beta * self.context_scale

        h_tilde = h * torch.sigmoid(gamma) + beta
        return self.base_head(h_tilde)

    # ---------------------------
    # Forward
    # ---------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        ctx_nodes: Optional[torch.Tensor] = None,  # requerido só se fusion_mode!='none'
    ) -> torch.Tensor:
        """
        Retorna logits por nó [N, num_classes] (sem ativação).
        """
        h = self.get_node_embeddings(x, edge_index)  # [N, D]

        if self.fusion_mode == "none":
            return self.base_head(h)

        # Requer contexto para 'concat'/'film'
        assert (
            ctx_nodes is not None
        ), "ctx_nodes deve ser fornecido para 'concat'/'film'."
        assert (
            ctx_nodes.dim() == 2
            and ctx_nodes.size(0) == h.size(0)
            and ctx_nodes.size(1) == self.context_dim
        ), f"ctx_nodes deve ser [N, {self.context_dim}]"

        if self.fusion_mode == "concat":
            return self.fuse_concat(h, ctx_nodes)
        else:
            return self.fuse_film(h, ctx_nodes)

    # ---------------------------
    # Utils
    # ---------------------------

    @torch.no_grad()
    def set_context_scale(self, value: float):
        """
        Ajusta a escala global do contexto (útil para annealing via trainer).
        """
        self.context_scale.fill_(float(value))
