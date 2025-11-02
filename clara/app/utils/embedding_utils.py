import numpy as np
from Bio.PDB import is_aa


def calculate_edge_embedding(emb1, emb2, method="absolute_difference"):
    """
    Calculate an edge embedding from two node embeddings.

    Args:
        emb1 (np.ndarray): First node embedding.
        emb2 (np.ndarray): Second node embedding.
        method (str): Method to calculate edge embedding. Options:
            - "absolute_difference": element-wise absolute difference
            - "difference": element-wise difference (emb1 - emb2)
            - "cosine_similarity": cosine similarity (single value)
            - "dot_product": dot product (single value)
            - "concat": concatenate emb1 and emb2
            - "average": element-wise average of emb1 and emb2

    Returns:
        np.ndarray: Computed edge embedding as a numpy array.
    """
    if method == "absolute_difference":
        return np.abs(emb1 - emb2)

    elif method == "difference":
        return emb1 - emb2

    elif method == "cosine_similarity":
        cos_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        return np.array([cos_sim])

    elif method == "dot_product":
        dot = np.dot(emb1, emb2)
        return np.array([dot])

    elif method == "concat":
        return np.concatenate([emb1, emb2])

    elif method == "average":
        return (emb1 + emb2) / 2

    else:
        raise ValueError(
            f"[ERROR] Unknown method {method} for calculating edge embeddings."
        )


def get_residue_id(residue):
    """
    Create a standardized residue identifier: RESNAME_RESNUM_CHAIN (e.g., LYS_12_A).

    Args:
        residue (Residue): A Biopython Residue object.

    Returns:
        str: Formatted residue identifier.
    """
    return f"{residue.get_resname()}_{residue.get_id()[1]}_{residue.get_parent().id}"


def extract_residue_ids(protein):
    """
    Internal function to extract residue IDs from the protein structure.

    Returns:
        list: List of residue identifiers (e.g., LYS_12_A).
    """
    residue_ids = []
    for model in protein.structure:
        for chain in model:
            if chain.id != protein.chain_id and protein.chain_id != "":
                continue
            for residue in chain:
                if residue.id[0] == " " and is_aa(residue):
                    res_name = residue.get_resname().strip()
                    res_num = residue.get_id()[1]
                    residue_id = f"{res_name}_{res_num}_{chain.id}"
                    residue_ids.append(residue_id)

    return residue_ids
