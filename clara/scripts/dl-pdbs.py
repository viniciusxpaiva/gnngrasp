import os
import pandas as pd


def load_biolip():
    # Load BioLiP file into a DataFrame
    biolip_path = "../data/BioLiP_nr.txt"
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


def download_pdb(pdb_id, pdb_dir):
    # Download PDB file if not present locally
    cmd = f"wget https://files.rcsb.org/view/{pdb_id}.pdb -P {pdb_dir}"
    os.system(cmd)


pdb_dir = "../data/pdbs"
biolip_df = load_biolip()
filtered_rows = []
not_pdb_rows = []

for i, (_, row) in enumerate(biolip_df.iterrows()):
    # A cada 1000 linhas, imprime progresso
    if i % 3 == 0:
        print(f"[INFO] Processed {i}/{len(biolip_df)} lines...")
    pdb_id = row["PDB_ID"].lower()
    chain_id = row["Receptor_chain"]
    key = f"{pdb_id}_{chain_id}"
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

    if not os.path.isfile(pdb_path):
        not_pdb_rows.append(row)
        # download_pdb(pdb_id, pdb_dir)
    else:
        filtered_rows.append(row)

# Salva apenas os PDBs encontrados com sucesso
filtered_df = pd.DataFrame(filtered_rows)
filtered_df.to_csv("biolip_templates_filtered.txt", sep="\t", index=False, header=False)

not_pdb_df = pd.DataFrame(not_pdb_rows)
not_pdb_df.to_csv("not_pdb_templates.txt", sep="\t", index=False, header=False)
