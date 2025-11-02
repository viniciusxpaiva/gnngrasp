import os
import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from app.utils.embedding_utils import extract_residue_ids


class ProtBertNodeEmbeddingsGenerator:
    """
    Generates node embeddings from protein sequences using ProtBert.
    """

    def __init__(self, output_dir="templates/node_embeddings"):
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ProtBert model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, sequence: str):
        """
        Prepare the protein sequence in ProtBert's expected format.
        e.g. M E T A ... instead of META...
        """
        # Insert spaces between each amino acid
        sequence = ' '.join(sequence)
        return self.tokenizer(sequence, return_tensors="pt")

    def generate_node_embeddings(self, protein):
        """
        Generate node embeddings using ProtBert.

        Args:
            protein (Protein): Protein object with .sequence, .pdb_id, .chain_id

        Returns:
            pd.DataFrame: Embeddings per residue
        """
        if not protein.is_valid:
            return

        residue_ids = extract_residue_ids(protein)
        if not residue_ids:
            return

        inputs = self._prepare_input(protein.sequence)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Sequence output: (1, seq_len, hidden_dim)
            embeddings = outputs.last_hidden_state[0]

        # Remove CLS and SEP
        residue_embeddings = embeddings[1 : len(protein.sequence) + 1].cpu().numpy()

        if len(residue_embeddings) != len(residue_ids):
            raise ValueError(
                f"[ERROR] Node count mismatch for {protein.pdb_id}_{protein.chain_id}: {len(residue_ids)} vs {len(residue_embeddings)}"
            )

        df = pd.DataFrame(residue_embeddings)
        df.insert(0, "residue_id", residue_ids)

        return df
