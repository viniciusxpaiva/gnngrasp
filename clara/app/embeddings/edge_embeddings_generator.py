import numpy as np
import pandas as pd
from app.utils.embedding_utils import calculate_edge_embedding, get_residue_id


class EdgeEmbeddingsGenerator:
    """
    Class responsible for generating edge information (connections)
    between residues in a protein structure, based purely on distance.
    """

    def __init__(self, distance_cutoff=5.0):
        """
        Initialize the EdgeEmbeddingGenerator.

        Args:
            distance_cutoff (float): Maximum distance (in Å) between residues to create an edge.
        """
        self.distance_cutoff = distance_cutoff

    def generate_edge_embeddings(self, protein):
        """
        Generate edge embeddings from node embeddings by taking the absolute difference.

        Args:
            protein (Protein): A protein object with node embeddings and structure information.

        Returns:
            pd.DataFrame: DataFrame containing source, target, and edge embeddings.
        """

        if not protein.is_valid:
            raise ValueError(
                f"[ERROR] Invalid protein structure: {protein.pdb_id}_{protein.chain_id}"
            )

        # Get node embeddings DataFrame from protein object
        node_embeddings_df = protein.node_embeddings

        # Dictionary mapping residue_id to its embedding for lookup
        embeddings_dict = {
            row["residue_id"]: row[1:].values.astype(float)
            for _, row in node_embeddings_df.iterrows()
        }
        # Embedding size validation
        embedding_size = node_embeddings_df.shape[1] - 1  # exclude residue_id

        # Extract all residues that have a C-alpha atom
        residues = [res for res in protein.structure.get_residues() if "CA" in res]

        edges = []

        # Loop through each unique residue pair
        for i, res1 in enumerate(residues):
            res1_id = get_residue_id(res1)
            emb1 = embeddings_dict.get(res1_id)

            # Skip residues without embeddings
            if emb1 is None:
                continue
            for j in range(i + 1, len(residues)):
                res2 = residues[j]
                res2_id = get_residue_id(res2)
                emb2 = embeddings_dict.get(res2_id)

                if emb2 is None:
                    continue

                # Calculate Euclidean distance between C-alpha atoms
                distance = np.linalg.norm(res1["CA"].coord - res2["CA"].coord)

                # Only create edges below the distance cutoff
                if distance <= self.distance_cutoff:
                    # Compute absolute difference embedding
                    # edge_embedding = calculate_edge_embedding(
                    #    emb1, emb2, edge_embed_method
                    # )

                    edge_data_forward = {"source": res1_id, "target": res2_id}
                    edge_data_reverse = {"source": res2_id, "target": res1_id}

                    # for idx in range(embedding_size):
                    #    edge_data_forward[f"{idx}"] = edge_embedding[idx]
                    #    edge_data_reverse[f"{idx}"] = edge_embedding[idx]

                    edges.extend([edge_data_forward, edge_data_reverse])
        return pd.DataFrame(edges)
