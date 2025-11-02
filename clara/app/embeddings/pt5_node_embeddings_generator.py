import torch
import re
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
from app.utils.embedding_utils import extract_residue_ids


class ProstT5NodeEmbeddingsGenerator:
    """
    Generates node embeddings from protein sequences using ProstT5.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/ProstT5", do_lower_case=False
        )
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)

        # Use half precision on GPU, full on CPU
        if self.device.type == "cpu":
            self.model.float()
        else:
            self.model.half()

        self.model.eval()

    def generate_node_embeddings(self, protein):
        """
        Generate node embeddings for the given protein using ProstT5.

        Args:
            protein (Protein): Protein object containing sequence and structure

        Returns:
            pd.DataFrame: DataFrame with residue_id and embeddings
        """
        if not protein.is_valid:
            return

        sequence = protein.sequence.upper()
        sequence = re.sub(r"[UZOB]", "X", sequence)  # replace rare AAs
        spaced_sequence = " ".join(list(sequence))
        prost_input = "<AA2fold> " + spaced_sequence

        encoded = self.tokenizer.batch_encode_plus(
            [prost_input],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=encoded.input_ids, attention_mask=encoded.attention_mask
            )

        # remove special token embeddings (CLS and SEP)
        sequence_len = len(sequence)
        embedding = output.last_hidden_state[
            0, 1 : 1 + sequence_len
        ]  # shape (L x 1024)
        embedding_np = embedding.cpu().numpy()

        residue_ids = extract_residue_ids(protein)

        if len(residue_ids) != len(embedding_np):
            raise ValueError(
                f"[ERROR] Node count mismatch for {protein.pdb_id}_{protein.chain_id}: "
                f"{len(residue_ids)} residues vs {len(embedding_np)} embeddings"
            )

        df = pd.DataFrame(embedding_np)
        df.insert(0, "residue_id", residue_ids)

        return df
