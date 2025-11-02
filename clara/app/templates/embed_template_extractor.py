import os
import pandas as pd
from Bio.PDB import is_aa
from app.utils.utils import download_pdb, three_to_one, load_biolip
from app.protein import Protein


class EmbedTemplateExtractor:
    """
    Class to extract templates from BioLiP dataset
    and generate residue-level and global-level embeddings.
    """

    def __init__(self, dirs):
        """
        Initialize the template extractor.

        Args:
            dirs (dict): Dictionary containing relevant directories and settings.
        """
        self.dirs = dirs
        self.biolip_df = load_biolip(self.dirs["data"]["biolip_dataset"])

    def generate_embeddings_templates(
        self,
        node_embedding_generator,  # ProtBertNodeEmbeddingsGenerator
    ):
        """
        Process all BioLiP entries and generate only node embeddings using ProtBert.
        """
        accumulator = {}

        print("[INFO] Processing BioLiP dataset (ESM)")

        # Step 1: Process each BioLiP row
        count = 0
        for _, row in self.biolip_df.iterrows():
            try:
                self.process_biolip_row(row, accumulator)
            except Exception as e:
                pdb_id, chain_id = row["PDB_ID"], row["Receptor_chain"]
                print(f"[ERROR] Failed to process {pdb_id}_{chain_id}: {e}")
                self.save_embedding_error_log(pdb_id, chain_id, e)
            if count % 1000 == 0:
                print(f"[!] Processed {count}/{len(self.biolip_df)} dataset rows.")
            count += 1

        print("[INFO] Starting node embedding generation")
        count = 0

        # Step 2: Process each PDB-chain collected
        for key, residues in accumulator.items():
            pdb_id, chain_id = key.split("_")
            pdb_path = os.path.join(self.dirs["data"]["pdbs"], f"{pdb_id}.pdb")

            if not os.path.isfile(pdb_path):
                print(f"[WARNING] PDB not found: {pdb_path}")
                download_pdb(pdb_id, self.dirs["data"]["pdbs"])

            if not os.path.isfile(pdb_path):
                print(f"[WARNING] PDB not found: {pdb_path}")
                continue

            try:
                protein = Protein(pdb_id, pdb_path, chain_id)

                if not protein.is_valid:
                    print(
                        f"[WARNING] Skipping {protein.pdb_id}_{protein.chain_id} (invalid structure)"
                    )
                    # self.save_invalid_sequence_log(pdb_id, chain_id)
                    continue

                # ➤ Only generate node embeddings with ProtBert
                protein.node_embeddings = (
                    node_embedding_generator.generate_node_embeddings(protein)
                )
                self.save_node_embeddings_to_csv(key, residues, protein.node_embeddings)

                if count % 1000 == 0:
                    print(
                        f"[!] Processed {count}/{len(accumulator)} templates successfully."
                    )
                count += 1

            except Exception as e:
                print(f"[ERROR] Error processing {pdb_id}_{chain_id}: {e}")
                self.save_embedding_error_log(pdb_id, chain_id, e)

        print("[INFO] Node embedding generation completed.")

    def process_biolip_row(self, row, accumulator):
        """
        Process a single BioLiP row to collect binding site labels.

        Args:
            row (pd.Series): Row from BioLiP dataset.
            accumulator (dict): Accumulated residues dictionary.
        """
        pdb_id = row["PDB_ID"].lower()
        chain_id = row["Receptor_chain"]
        key = f"{pdb_id}_{chain_id}"

        pdb_path = os.path.join(self.dirs["data"]["pdbs"], f"{pdb_id}.pdb")
        if not self.verify_pdb_existence(pdb_id, chain_id, pdb_path):
            return

        # Extract annotated binding residues
        binding_residues = row["Binding_site_residues_PDB"].split()
        bs_set = set(binding_residues)

        # Initialize Protein
        protein = Protein(pdb_id, pdb_path, chain_id)
        if not protein.is_valid or not protein.get_chains():
            print(f"[ERROR] Structure loading failed: {pdb_id}_{chain_id}")
            with open("missing_chains.log", "a") as f:
                f.write(f"{pdb_id}_{chain_id}\n")
            return

        if key not in accumulator:
            accumulator[key] = {}

            for chain in protein.get_chains():
                for residue in chain:
                    if residue.id[0] != " " or not is_aa(residue):
                        continue
                    resname = residue.resname.strip()
                    resnum = residue.id[1]
                    residue_id = f"{resname}_{resnum}_{chain.id}"
                    accumulator[key][residue_id] = 0  # Non-binding by default

        # Update binding site labels
        self.assign_binding_labels(accumulator[key], bs_set)

    def assign_binding_labels(self, residue_dict, bs_set):
        """
        Assign binding site labels to residues.

        Args:
            residue_dict (dict): Mapping residue_id → label.
            bs_set (set): Set of binding site residue short names from BioLiP.
        """
        for residue_id in residue_dict:
            resname, resnum, _ = residue_id.split("_")
            one_letter = three_to_one(resname)
            res_short = f"{one_letter}{resnum}"
            if res_short in bs_set:
                residue_dict[residue_id] = 1

    def save_node_embeddings_to_csv(self, key, residues, node_embeddings):
        """
        Save node embeddings merged with binding labels to CSV.

        Args:
            key (str): Identifier pdbid_chain.
            residues (dict): Dictionary mapping residue_id → label.
            node_embeddings (DataFrame): DataFrame of node embeddings.
        """
        labels_df = pd.DataFrame(
            list(residues.items()), columns=["residue_id", "label"]
        )
        final_df = pd.merge(labels_df, node_embeddings, on="residue_id", how="inner")

        # Reorder columns
        cols = ["residue_id", "label"] + [
            col for col in final_df.columns if col not in ["residue_id", "label"]
        ]
        final_df = final_df[cols]

        output_path = os.path.join(
            self.dirs["data"]["esm_templates"]["node_embeddings"],
            f"{key}_node_embeddings.csv.zip",
        )
        final_df.to_csv(output_path, index=False, compression="zip")

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
            f"{key}_edge_embeddings.csv.zip",
        )

        # Save the DataFrame
        edge_embeddings.to_csv(
            edge_embeddings_output_path, index=False, compression="zip"
        )

        # print(f"[+] Saved edge embeddings for {key} at {edge_embeddings_output_path}")

    def save_global_embeddings_to_csv(self, global_embeddings_list):
        """
        Save all global embeddings into a single CSV.

        Args:
            global_embeddings_list (list): List of individual global embeddings (DataFrames).
        """
        if not global_embeddings_list:
            print("[WARNING] No global embeddings to save.")
            return

        final_global_embeddings = pd.concat(global_embeddings_list, ignore_index=True)
        cols = ["pdb_id"] + [
            col for col in final_global_embeddings.columns if col != "pdb_id"
        ]
        final_global_embeddings = final_global_embeddings[cols]

        output_path = os.path.join(
            self.dirs["data"]["embd_templates"]["global_embeddings"],
            "global_embeddings_templates.csv.zip",
        )
        final_global_embeddings.to_csv(output_path, index=False, compression="zip")
        print(f"[INFO] Global embeddings saved at: {output_path}")

    #############################################################
    # AUXILIARY FUNCTIONS
    #############################################################

    def is_valid_protein(self, protein):
        """
        Validate a protein sequence.
        """
        return bool(protein.sequence) and set(protein.sequence) != {"X"}

    def verify_pdb_existence(self, pdb_id, chain_id, pdb_path):
        """
        Check if a PDB file exists locally.
        """
        if not os.path.isfile(pdb_path):
            print(f"[WARNING] Missing PDB file: {pdb_id}")
            with open("missing_pdbs.log", "a") as f:
                f.write(f"{pdb_id}_{chain_id}\n")
            return False
        return True

    def save_invalid_sequence_log(self, pdb_id, chain_id):
        """
        Save log for invalid protein sequences.
        """
        with open("invalid_sequences.log", "a") as f:
            f.write(f"{pdb_id}_{chain_id}\n")

    def save_embedding_error_log(self, pdb_id, chain_id, error):
        """
        Save log for embedding generation errors.
        """
        with open("embedding_errors.log", "a") as f:
            f.write(f"{pdb_id}_{chain_id} - {error}\n")
