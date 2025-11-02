import torch
import torch.nn as nn


class AttentionFeatures(nn.Module):
    """
    A module that combines node embeddings and handcrafted features
    using optional attention, or projects directly if only one input exists.
    """

    def __init__(
        self,
        embd_dim=1280,
        prop_dim=20,
        proj_dim=128,
        num_heads=4,
        use_embd=True,
        use_prop=True,
    ):
        """
        Initialize the AttentionFeatures module.

        Args:
            embd_dim (int): Dimension of the input embeddings.
            prop_dim (int): Dimension of the handcrafted features.
            proj_dim (int): Projection dimension to hidden size.
            num_heads (int): Number of attention heads (only if both inputs are used).
            use_embd (bool): Whether embeddings are used.
            use_prop (bool): Whether properties are used.
        """
        super().__init__()

        self.use_embd = use_embd
        self.use_prop = use_prop
        self.proj_dim = proj_dim

        # If using both, create projection + attention
        self.apply_attention = use_embd and use_prop

        if self.apply_attention:
            self.embd_proj = nn.Linear(embd_dim, proj_dim)
            self.prop_proj = nn.Linear(prop_dim, proj_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=num_heads,
                batch_first=True,
            )
        else:
            # If using only embeddings or only properties, still need projection
            if use_embd and embd_dim is not None:
                self.embd_proj = nn.Linear(embd_dim, proj_dim)

            if use_prop and prop_dim is not None:
                self.prop_proj = nn.Linear(prop_dim, proj_dim)

    def forward(self, embd=None, prop=None):
        """
        Forward pass to combine inputs.

        Args:
            embd (Tensor or None): Embedding tensor.
            prop (Tensor or None): Property tensor.

        Returns:
            Tensor: Output features projected to proj_dim.
        """

        if self.apply_attention:
            embd_proj = self.embd_proj(embd)
            prop_proj = self.prop_proj(prop)

            attn_output, _ = self.attention(
                query=embd_proj.unsqueeze(0),
                key=prop_proj.unsqueeze(0),
                value=prop_proj.unsqueeze(0),
            )
            combined_features = embd_proj + attn_output.squeeze(0)
            return combined_features

        elif embd is not None:
            return self.embd_proj(embd)  # Project embeddings

        elif prop is not None:
            return self.prop_proj(prop)  # Project properties

        else:
            raise ValueError(
                "At least one of embeddings or properties must be provided."
            )

    def summary(self):
        """
        Print a summary of how the AttentionFeatures module will behave.
        """
        if self.use_embd and self.use_prop:
            print(
                "[ATTENTION] Mode: embeddings + properties → Using attention mechanism."
            )
        elif self.use_embd:
            print("[ATTENTION] Mode: only embeddings → No attention applied.")
        elif self.use_prop:
            print("[ATTENTION] Mode: only properties → No attention applied.")
        else:
            print(
                "[ATTENTION] Invalid configuration: neither embeddings nor properties available."
            )
