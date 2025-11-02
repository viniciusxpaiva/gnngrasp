import os
import numpy as np
import pandas as pd
from app.protein import Protein
from app.utils.embedding_utils import calculate_edge_embedding, get_residue_id


class EdgeDistanceGenerator:
    """
    Class responsible for generating edge information (connections)
    between residues in a protein structure, based purely on distance.
    """

    def __init__(self, pdb_id, pdb_path, chain_id, distance_cutoff):
        """
        Initialize the EdgeEmbeddingGenerator.

        Args:
            distance_cutoff (float): Maximum distance (in Å) between residues to create an edge.
        """
        self.distance_cutoff = distance_cutoff
        self.protein = Protein(pdb_id, pdb_path, chain_id)
        self.node_embd_path = f"/home/vinicius/Desktop/deep-grasp/data/esm_templates/node_embeddings/{pdb_id}_{chain_id}_node_embeddings.csv.zip"

    def generate_edge_embeddings(self):
        """
        Generate edge embeddings from node embeddings by taking the absolute difference.

        Args:
            protein (Protein): A protein object with node embeddings and structure information.

        Returns:
            pd.DataFrame: DataFrame containing source, target, and edge embeddings.
        """

        if not self.protein.is_valid:
            raise ValueError(
                f"[ERROR] Invalid protein structure: {self.protein.pdb_id}_{self.protein.chain_id}"
            )

        # Get node embeddings DataFrame from protein object
        node_embeddings_df = pd.read_csv(self.node_embd_path, compression="zip")

        # Dictionary mapping residue_id to its embedding for lookup
        embeddings_dict = {
            row["residue_id"]: row[1:].values.astype(float)
            for _, row in node_embeddings_df.iterrows()
        }

        # Extract all residues that have a C-alpha atom
        residues = [res for res in self.protein.structure.get_residues() if "CA" in res]

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
                    edge_data_forward = {
                        "source": res1_id,
                        "target": res2_id,
                        "distance": distance,
                    }
                    edge_data_reverse = {
                        "source": res2_id,
                        "target": res1_id,
                        "distance": distance,
                    }

                    edges.extend([edge_data_forward, edge_data_reverse])
        return pd.DataFrame(edges)

    def save_edge_embeddings_to_csv(self, key, edge_embeddings):
        """
        Save edge embeddings (basic connections) to a CSV file.

        Args:
            key (str): Identifier for the template (e.g., pdbid_chain).
            edge_embeddings (DataFrame): DataFrame containing edge information (source, target).
        """
        # Reorder columns to have a standard format
        cols = ["source", "target"] + [
            col for col in edge_embeddings.columns if col not in ["source", "target"]
        ]
        edge_embeddings = edge_embeddings[cols]

        # Define output path
        edge_embeddings_output_path = os.path.join(
            self.dirs["data"]["pbert_embeddings"]["edge_embeddings"],
            f"{key}_edge_distance.csv.zip",
        )

        # Save the DataFrame
        edge_embeddings.to_csv(
            edge_embeddings_output_path, index=False, compression="zip"
        )

        # print(f"[+] Saved edge embeddings for {key} at {edge_embeddings_output_path}")
