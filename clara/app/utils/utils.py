from Bio.PDB import PDBParser, PPBuilder
from Bio.Data.IUPACData import protein_letters_3to1, protein_letters_1to3
import os, requests
import pandas as pd

atom_types_file = "/home/vinicius/Desktop/deep-grasp/data/atom_types.txt"
# atom_types_file = "/storage/hpc/11/dealmei1/deep-grasp/data/atom_types.txt"


def pdb_to_sequence(pdb_file, chain_id=None):
    # Extract amino acid sequence from a PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ppb = PPBuilder()

    model = structure[0]  # Get the first model

    if chain_id is None:
        chain_id = next(model.get_chains()).id  # Default to the first chain

    sequence = ""
    for pp in ppb.build_peptides(model):
        sequence += pp.get_sequence()  # Concatenate peptide sequences

    if str(sequence) == "":
        print("Sequence was not obtained")
    return str(sequence)


def sequence_from_structure(structure, chain_id):

    aa_map = {k.upper(): v for k, v in protein_letters_3to1.items()}
    sequence = ""

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] == " ":
                    resname = residue.get_resname().strip().upper()
                    sequence += aa_map.get(resname, "X")  # Unknown residues as X
            return sequence


def load_biolip(biolip_path):
    # Load BioLiP file into a DataFrame
    print(f"Loading: {biolip_path}")
    biolip_df = pd.read_csv(biolip_path, sep="\t", header=None)
    biolip_df.columns = [
        "PDB_ID",
        "Receptor_chain",
        "Resolution",
        "Binding_site_number",
        "Ligand_ID",
        "Ligand_chain",
        "Ligand_serial_number",
        "Binding_site_residues_PDB",
        "Binding_site_residues_reindexed",
        "Catalytic_site_residues_PDB",
        "Catalytic_site_residues_reindexed",
        "EC_number",
        "GO_terms",
        "Affinity_manual",
        "Affinity_Binding_MOAD",
        "Affinity_PDBbindCN",
        "Affinity_BindingDB",
        "UniProt_ID",
        "PubMed_ID",
        "Ligand_residue_seq_number",
        "Receptor_sequence",
    ]
    return biolip_df


def download_pdb(pdb_id: str, save_dir: str):
    """
    Downloads a PDB file from the RCSB PDB website if it does not already exist locally.

    Args:
        pdb_id (str): The 4-character PDB identifier.
        save_dir (str): Directory where the PDB file should be saved.
    """
    pdb_id = pdb_id.lower()
    pdb_path = os.path.join(save_dir, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"[INFO] Downloading {pdb_id}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(pdb_path, "w") as f:
                f.write(response.text)
            print(f"[SUCCESS] {pdb_id} saved to {pdb_path}")
        else:
            print(f"[ERROR] Failed to download {pdb_id} (HTTP {response.status_code})")
    return pdb_path


def three_to_one(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except KeyError:
        return "X"


def one_to_three(self, resname):
    # Convert three-letter residue name to one-letter code
    try:
        return protein_letters_1to3[resname.capitalize()]
    except KeyError:
        return "X"  # Return X if conversion is not possible


def build_residue_id(residue):
    # Return residue ID in the format RESNAME_NUM_CHAIN (e.g., LYS_23_A)
    return f"{residue.get_resname().strip()}_{residue.get_id()[1]}_{residue.get_parent().id}"


def load_atom_types(atom_types_dict):
    # Load atom types from the atom_types.txt file
    df = pd.read_csv(atom_types_file, header=None)

    # Assign column names dynamically based on number of columns
    df.columns = ["Residue", "Atom", "Type1", "Type2", "Type3", "Type4"][
        : len(df.columns)
    ]

    # Build atom type dictionary: (residue, atom) → list of types
    for _, row in df.iterrows():
        residue, atom = row["Residue"], row["Atom"]
        types = row[2:].dropna().tolist()
        atom_types_dict[(residue, atom)] = types
