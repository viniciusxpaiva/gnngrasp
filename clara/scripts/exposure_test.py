from Bio.PDB import HSExposure, PDBParser
import pandas as pd
import os


def get_exposure_features(structure):
    """
    Calculates Half Sphere Exposure (HSE) features using HSExposureCB.
    If an error occurs (e.g., due to missing CB atoms), returns an empty DataFrame.
    """
    residue_data = []

    try:
        # May fail if some residues are malformed or missing CB atoms
        exp_cb = HSExposure.HSExposureCB(structure[0])
    except Exception as e:
        print(f"[!] Failed to compute HSExposure: {e}")
        return pd.DataFrame()

    for key in exp_cb.keys():
        res_chain = key[0]
        res_number = key[1][1]

        # Attempt to get residue; skip if not found
        try:
            residue = structure[0][res_chain][res_number]
            res_name = residue.get_resname()
        except KeyError:
            continue

        residue_id = f"{res_name}_{res_number}_{res_chain}"
        hse_values = exp_cb[key]

        residue_data.append(
            {
                "residue_id": residue_id,
                "hse_up": hse_values[0],
                "hse_down": hse_values[1],
            }
        )

    return pd.DataFrame(residue_data)


pdb_file = "../data/pdbs/1peg.pdb"
parser = PDBParser(QUIET=True)
pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
structure = parser.get_structure(pdb_id, pdb_file)
df = get_exposure_features(structure)
print(df)
