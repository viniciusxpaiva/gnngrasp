from Bio.PDB import NeighborSearch, Selection, is_aa
from app.utils.utils import load_atom_types
import pandas as pd
import numpy as np


class EdgePropertiesExtractor:
    """
    Extracts residue-residue edges based on the presence of real atomic interactions.
    An edge exists only if at least one interaction (e.g., hydrogen bond) is detected,
    regardless of the distance between C-alpha atoms.
    """

    def __init__(self, protein):
        """
        Initialize with a protein structure and interaction rules.
        """
        self.protein = protein
        self.atom_types_dict = {}
        load_atom_types(self.atom_types_dict)

    def get_edge_features(self):
        """
        Build a DataFrame with residue-residue edges based on detected interactions.

        Returns:
            pd.DataFrame: Each row represents an edge with interaction counts.
        """

        residues = [
            res
            for res in self.protein.structure.get_residues()
            if is_aa(res) and res.id[0] == " " and "CA" in res
        ]
        ns = NeighborSearch(
            list(Selection.unfold_entities(self.protein.structure, "A"))
        )

        edges = []

        for i, res1 in enumerate(residues):
            for j in range(i + 1, len(residues)):
                res2 = residues[j]

                res1_id = self._get_residue_id(res1)
                res2_id = self._get_residue_id(res2)

                ca_distance = np.linalg.norm(res1["CA"].coord - res2["CA"].coord)

                interactions = {
                    "aromatic_bond": 0,
                    "hydrogen_bond": 0,
                    "hydrophobic_bond": 0,
                    "salt_bridge": 0,
                    "repulsive_bond": 0,
                    "disulfide_bridge": 0,
                }

                for atom1 in res1:
                    neighbors = ns.search(atom1.coord, 6.5, "A")
                    for atom2 in neighbors:
                        if atom2.get_parent() != res2:
                            continue

                        interaction_type = self.get_interaction_type(atom1, atom2)
                        if interaction_type:
                            interactions[interaction_type] += 1

                if any(count > 0 for count in interactions.values()):
                    edges.append(
                        {
                            "source": res1_id,
                            "target": res2_id,
                            "distance": ca_distance,
                            **interactions,
                        }
                    )
                    edges.append(
                        {
                            "source": res2_id,
                            "target": res1_id,
                            "distance": ca_distance,
                            **interactions,
                        }
                    )

        return pd.DataFrame(edges)

    def get_edge_features_distance(self, cutoff_distance=8):
        """
        Build a DataFrame with residue-residue edges based on detected interactions.

        Returns:
            pd.DataFrame: Each row represents an edge with interaction counts.
        """

        residues = [
            res
            for res in self.protein.structure.get_residues()
            if is_aa(res) and res.id[0] == " " and "CA" in res
        ]

        edges = []

        for i, res1 in enumerate(residues):
            for j in range(i + 1, len(residues)):
                res2 = residues[j]

                res1_id = self._get_residue_id(res1)
                res2_id = self._get_residue_id(res2)

                distance = np.linalg.norm(res1["CA"].coord - res2["CA"].coord)

                if distance <= cutoff_distance:
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

    def get_interaction_type(self, atom1, atom2):
        """
        Determine interaction type between two atoms based on chemistry and distance.
        """
        types1 = self.atom_types_dict.get(
            (atom1.get_parent().get_resname(), atom1.get_name().strip()), []
        )
        types2 = self.atom_types_dict.get(
            (atom2.get_parent().get_resname(), atom2.get_name().strip()), []
        )

        distance = np.linalg.norm(atom1.coord - atom2.coord)

        if "ARM" in types1 and "ARM" in types2 and 1.5 <= distance <= 3.5:
            return "aromatic_bond"
        if (
            ("ACP" in types1 and "DON" in types2)
            or ("DON" in types1 and "ACP" in types2)
        ) and 2.0 <= distance <= 3.0:
            return "hydrogen_bond"
        if "HPB" in types1 and "HPB" in types2 and 2.0 <= distance <= 3.8:
            return "hydrophobic_bond"
        if (
            ("POS" in types1 and "NEG" in types2)
            or ("NEG" in types1 and "POS" in types2)
        ) and 2.0 <= distance <= 6.0:
            return "salt_bridge"
        if (
            ("POS" in types1 and "POS" in types2)
            or ("NEG" in types1 and "NEG" in types2)
        ) and 2.0 <= distance <= 6.0:
            return "repulsive_bond"
        if atom1.element == "S" and atom2.element == "S" and 2.0 <= distance <= 2.2:
            return "disulfide_bridge"

        return None

    def _get_residue_id(self, residue):
        """
        Build a unique residue ID string in the format: RESNAME_RESNUM_CHAIN
        """
        return (
            f"{residue.get_resname()}_{residue.get_id()[1]}_{residue.get_parent().id}"
        )
