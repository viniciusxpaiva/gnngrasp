import pandas as pd
import os


def load_biolip(biolip_path):
    # Load BioLiP file into a DataFrame
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


def get_biolip_unique_sites():

    df = load_biolip("/home/vinicius/Desktop/deep-bender/data/BioLiP.txt")
    total_bs = list(df["PDB_ID"])
    unique_sites = list(
        (df["PDB_ID"].str.lower() + df["Receptor_chain"].str.lower()).unique()
    )
    return len(total_bs), (unique_sites)


def get_grasp_unique_sites():
    grasp_tpt_dir = "/home/vinicius/Desktop/dl-grasp/app/data/templates"
    tpt_files = os.listdir(grasp_tpt_dir)
    sem_cvs_zip = [file[:5].lower() for file in tpt_files]
    return sem_cvs_zip


total_bs, unique_sites = get_biolip_unique_sites()
print(f"Total binding sites biolip: {total_bs} | Unique: {len(unique_sites)}")

total_grasp = get_grasp_unique_sites()
print(f"Total grasp templates: {len(total_grasp)}")

unique_sites_set = set(unique_sites)
total_grasp_set = set(total_grasp)


print(unique_sites_set)
grasp_only = total_grasp_set - unique_sites_set
grasp_only_series = pd.Series(sorted(grasp_only))
grasp_only_series.to_csv("grasp_only_templates.txt", index=False, header=False)

print(grasp_only_series)
print(grasp_only_series.sample(10))


common_templates = unique_sites_set & total_grasp_set

# Converte para Series
common_templates_series = pd.Series(sorted(common_templates))
common_templates_series.to_csv("common_templates.txt", index=False, header=False)
