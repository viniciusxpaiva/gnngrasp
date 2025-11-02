import pandas as pd
from Bio.PDB import is_aa


class ResidueNeighborsExtractor:
    """
    Extracts direct neighbors (interacting residues) for each residue in a protein structure.
    Saves the result as a CSV file: each row contains a residue and its directly interacting neighbors.
    """

    def __init__(self, protein, edge_extractor):
        """
        Args:
            protein: Protein object with .structure (Bio.PDB structure).
            edge_extractor: Instance of an edge extraction class (e.g., EdgePropertiesExtractor).
        """
        self.protein = protein
        self.edge_extractor = edge_extractor

    def _get_all_residues(self):
        """
        Returns a list of all valid residues (amino acids with C-alpha atoms).
        """
        residues = [
            res
            for res in self.protein.structure.get_residues()
            if is_aa(res) and res.id[0] == " " and "CA" in res
        ]
        return residues

    def _get_residue_ids(self, residues):
        """
        Returns a list of unique residue IDs for a list of residues.
        """
        return [self.edge_extractor._get_residue_id(res) for res in residues]

    def _build_neighbors_dict(self, edge_df, residue_ids):
        """
        Constructs a dictionary mapping each residue ID to a list of its interacting neighbors.
        """
        # Initialize empty neighbor lists for each residue
        neighbors_dict = {rid: [] for rid in residue_ids}

        # Define interaction types to consider
        interaction_types = [
            "aromatic_bond",
            "hydrogen_bond",
            "hydrophobic_bond",
            "salt_bridge",
            "repulsive_bond",
            "disulfide_bridge",
        ]

        # Populate neighbor lists from edge DataFrame
        for _, row in edge_df.iterrows():
            if any(row[itype] > 0 for itype in interaction_types):
                neighbors_dict[row["source"]].append(row["target"])

        return neighbors_dict

    def extract_neighbors(self):
        """
        Main method to extract neighbors and return a DataFrame.
        Each row: residue_id, neighbors (comma-separated).
        """
        # Step 1: Get all valid residues and their IDs
        residues = self._get_all_residues()
        residue_ids = self._get_residue_ids(residues)

        # Step 2: Extract all edges (interactions)
        edge_df = self.edge_extractor.get_edge_features()

        # Step 3: Build neighbors dictionary
        neighbors_dict = self._build_neighbors_dict(edge_df, residue_ids)

        # Step 4: Create DataFrame with residue_id and its neighbors
        data = []
        for rid in residue_ids:
            data.append(
                {
                    "residue_id": rid,
                    "neighbors": ",".join(neighbors_dict[rid]),
                    # Add other fields if needed, e.g., labels, features
                }
            )

        return pd.DataFrame(data)

    def save_to_csv(self, csv_path):
        """
        Extracts neighbors and saves the resulting DataFrame to a CSV file.

        Args:
            csv_path (str): Output path for the CSV file.
        """
        df = self.extract_neighbors()
        df.to_csv(csv_path, index=False)
        return df
