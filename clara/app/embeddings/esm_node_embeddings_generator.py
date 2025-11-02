import os
import torch
import esm  # Meta AI's Evolutionary Scale Modeling library
import pandas as pd
from app.utils.embedding_utils import extract_residue_ids


class ESMNodeEmbeddingsGenerator:
    """
    Class responsible for generating node-level (residue) embeddings
    from a Protein object using the pretrained ESM model.
    """

    def __init__(
        self,
        output_dir="templates/node_embeddings",
        model_name="esm2_t33_650M_UR50D",
    ):
        """
        Initialize the NodeEmbeddingGenerator.

        Args:
            output_dir (str): Directory where embeddings will be saved.
            model_name (str): Pretrained ESM model to use.
        """
        self.output_dir = output_dir

        # Load pretrained ESM model and alphabet
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()

        self.model.eval()  # Set model to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_node_embeddings(self, protein, layer=33):
        """
        Generate node embeddings for a given protein structure.

        Args:
            protein (Protein): A protein object containing sequence and structure.
            layer (int): ESM layer from which to extract the embeddings.

        Returns:
            pd.DataFrame: DataFrame containing residue_id and corresponding embeddings.
        """

        # Validate the protein sequence
        if not protein.is_valid:
            return

        # Prepare the input for ESM model
        batch_labels, _, batch_tokens = self.batch_converter(
            [(protein.pdb_id, protein.sequence)]
        )
        batch_tokens = batch_tokens.to(self.device)

        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer])

        token_embeddings = results["representations"][layer][0]

        # Remove special tokens (BOS, EOS)
        node_embeddings = token_embeddings[1 : len(protein.sequence) + 1].cpu().numpy()

        # Match embeddings with correct residue_ids
        residue_ids = extract_residue_ids(protein)

        # Check consistency
        if len(residue_ids) != len(node_embeddings):
            raise ValueError(
                f"[ERROR] Node count mismatch for {protein.pdb_id}_{protein.chain_id}: {len(residue_ids)} vs {len(node_embeddings)}"
            )

        # Build the output DataFrame
        embedding_df = pd.DataFrame(node_embeddings)
        embedding_df.insert(0, "residue_id", residue_ids)

        return embedding_df
