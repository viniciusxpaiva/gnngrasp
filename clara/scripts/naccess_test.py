import os
import pandas as pd
import shutil
from Bio.PDB import PDBParser, is_aa, NACCESS


def run_naccess_and_parse_rsa(pdb_file, naccess_binary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    base_tmp_dir = os.path.join(output_dir, f"tmp_{pdb_id}")
    os.makedirs(base_tmp_dir, exist_ok=True)

    tmp_pdb_path = os.path.join(base_tmp_dir, f"{pdb_id}.pdb")
    shutil.copy(pdb_file, tmp_pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, tmp_pdb_path)

    # Run NACCESS (creates internal tmp directory inside base_tmp_dir)
    _ = NACCESS.NACCESS(
        model=structure,
        pdb_file=tmp_pdb_path,
        naccess_binary=naccess_binary,
        tmp_directory=base_tmp_dir,
    )

    # Find subdirectory created inside base_tmp_dir
    internal_tmp_dirs = [
        os.path.join(base_tmp_dir, d)
        for d in os.listdir(base_tmp_dir)
        if os.path.isdir(os.path.join(base_tmp_dir, d)) and d.startswith("tmp")
    ]

    if not internal_tmp_dirs:
        raise FileNotFoundError("No internal NACCESS tmp directory found.")

    # Assume only one (or take the most recent)
    internal_tmp_dirs.sort(key=os.path.getmtime, reverse=True)
    internal_dir = internal_tmp_dirs[0]

    # Find the .rsa file (any file ending in .rsa)
    rsa_files = [f for f in os.listdir(internal_dir) if f.endswith(".rsa")]
    if not rsa_files:
        raise FileNotFoundError(
            "No .rsa file found inside internal NACCESS tmp directory."
        )

    rsa_path = os.path.join(internal_dir, rsa_files[0])

    # Parse .rsa file
    residue_data = []
    with open(rsa_path) as f:
        for line in f:
            if not line.startswith("RES"):
                continue
            resname = line[4:7].strip()
            chain = line[8].strip()
            resnum = line[9:13].strip()
            residue_id = f"{resname}_{resnum}_{chain}"
            residue_data.append(
                {
                    "residue_id": residue_id,
                    "acc_all_rel": float(line[22:28].strip()),
                    "acc_main_rel": float(line[48:54].strip()),
                    "acc_side_rel": float(line[35:41].strip()),
                    "acc_polar_rel": float(line[75:81].strip()),
                    "acc_apolar_rel": float(line[61:67].strip()),
                }
            )

    return pd.DataFrame(residue_data)


pdb_file = "../data/pdbs/1a8t.pdb"
naccess_binary = "/home/vinicius/naccess/naccess"
output_dir = "../naccess_output/"

df = run_naccess_and_parse_rsa(pdb_file, naccess_binary, output_dir)
print(df.head())
