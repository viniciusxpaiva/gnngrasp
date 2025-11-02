import torch
import esm
import pandas as pd


class GlobalEmbeddingsGenerator:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device=None):
        """
        Initialize ESM-2 model for embedding generation.

        Args:
            model_name (str): Pre-trained model from ESM.
            device (str or torch.device): Device to run inference (default: CPU or CUDA if available).
        """
        self.model_name = model_name
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def generate_global_embeddings(self, protein, layer=12):
        """
        Generate global embedding for a single Protein object.

        Args:
            protein (Protein): A Protein object with .pdb_id and .sequence attributes.
            layer (int): ESM layer to extract embeddings from.

        Returns:
            DataFrame: A single-row DataFrame with the embedding.
        """

        # Prepare input in the format expected by the batch converter
        batch_labels, _, batch_tokens = self.batch_converter(
            [(protein.pdb_id, protein.sequence)]
        )
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer])

        token_representations = results["representations"][layer]

        # Calculate mean embedding (excluding special tokens)
        seq_len = batch_lens[0]
        embedding = token_representations[0, 1 : seq_len - 1].mean(0).cpu().numpy()

        # Return as DataFrame with the protein ID as index
        embedding_df = pd.DataFrame([embedding])
        embedding_df.insert(0, "pdb_id", f"{protein.pdb_id}_{protein.chain_id}")
        return embedding_df
