import os
import pandas as pd
from Bio.PDB import PDBParser, is_aa
from app.utils.utils import download_pdb, three_to_one, load_biolip
from app.protein import Protein
from app.properties.edge_properties_extractor import EdgePropertiesExtractor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


class PropTemplateExtractor:
    def __init__(self, dirs):
        # Initialize extractor with paths and load BioLiP data
        self.dirs = dirs
        self.biolip_df = load_biolip(self.dirs["data"]["biolip_dataset"])

    def generate_template_features(self):
        accumulator = {}

        print("Processing Biolip dataset")
        # Step 1: Process all BioLiP entries and collect labeled residues
        count = 0
        for _, row in self.biolip_df.iterrows():
            try:
                self.process_biolip_row(row, accumulator)
            except Exception as e:
                print(
                    f"[x] Failed to process row: {row['PDB_ID']}_{row['Receptor_chain']} - {e}"
                )
                with open("processing_errors.log", "a") as f:
                    f.write(f"{row['PDB_ID']}_{row['Receptor_chain']} - {e}\n")
            if count % 1000 == 0:
                print(f"[✓] Processed {count}/{len(self.biolip_df)} dataset rows.")
            count += 1

        # records = []
        # fasta_output_path = os.path.join(self.dirs["data"]["blast"], "biolip.fasta")

        print("Starting node and features extraction")
        # Step 2: For each PDB-chain, extract and save node and edge features
        count = 0
        for key, _ in accumulator.items():
            # print(f"Processing template: {key}")
            pdb_id, chain_id = key.split("_")
            pdb_path = (
                f"{self.dirs["data"]["pdbs"]}/{pdb_id}.pdb"  # Adjust path if needed
            )

            if not os.path.isfile(pdb_path):
                print(f"[!] PDB file not found: {pdb_path}")
                continue

            # Instantiate Protein class
            try:
                protein = Protein(pdb_id, pdb_path, chain_id)
                self.edge_prop_extractor = EdgePropertiesExtractor(protein)
                protein.edge_properties = self.edge_prop_extractor.get_edge_features()

                """
                if protein.sequence:
                    try:
                        record = SeqRecord(
                            Seq(protein.sequence),
                            id=key,  # formato "1abc_A"
                            description="",
                        )
                        records.append(record)
                    except Exception as e:
                        print(f"[x] Failed to create fasta record for {key}: {e}")
                """

            except ValueError as e:
                print(f"[!] Skipping protein {key}: {e}")
                with open("processing_errors.log", "a") as f:
                    f.write(f"{key} - {e}\n")
                continue

            # Check if node_features_df is valid (non-empty)
            if protein.edge_properties.empty:
                continue

            # Save node and edge features to CSV using dedicated function
            self.save_edge_features_to_csv(key, protein.edge_properties)

            if count % 1000 == 0:
                print(
                    f"[✓] Processed {count}/{len(accumulator)} templates successfully."
                )
            count += 1
        # Write all sequences to single .fasta file
        # if records:
        #    SeqIO.write(records, fasta_output_path, "fasta")
        #    print(f"[✓] Saved {len(records)} sequences to {fasta_output_path}")
        # else:
        #    print("[!] No sequences were saved to the fasta file.")

    def process_biolip_row(self, row, accumulator):
        """
        Processes a single entry (row) from BioLiP dataset, marking residues as binding or non-binding.

        Parameters:
            row (dict): Entry containing PDB data and binding residues from BioLiP.
            accumulator (dict): Accumulates residue information per PDB-chain key.
        """
        pdb_id = row["PDB_ID"].lower()
        chain_id = row["Receptor_chain"]
        key = f"{pdb_id}_{chain_id}"

        pdb_path = os.path.join(self.dirs["data"]["pdbs"], f"{pdb_id}.pdb")

        if not self.verify_pdb_file_existance(pdb_id, chain_id, pdb_path):
            return

        if os.path.isfile(
            f"/home/vinicius/Desktop/deep-grasp/data/prop_templates/edge_properties/{key}_edge_properties.csv.zip"
        ):
            return

        # Extract BioLiP annotated binding residues
        # binding_residues = row["Binding_site_residues_PDB"].split()
        # bs_set = set(binding_residues)

        # If this PDB-chain hasn't been processed yet, initialize Protein class
        if key not in accumulator:
            accumulator[key] = {}

            protein = Protein(pdb_id, pdb_path, chain_id)
            if protein.structure == None or protein.get_chains() == []:
                print(f"[x] Error loading structure {pdb_id}_{chain_id}")
                with open("missing_chains.txt", "a") as f:
                    f.write(f"{pdb_id}_{chain_id} - chain not found\n")
                return

            for chain in protein.get_chains():
                for residue in chain:
                    # Skip heteroatoms and non-amino acids
                    if residue.id[0] != " " or not is_aa(residue):
                        continue

                    resname = residue.resname.strip()
                    resnum = residue.id[1]
                    residue_id = f"{resname}_{resnum}_{chain.id}"

                    # Initialize residue as non-binding (0)
                    accumulator[key][residue_id] = 0

        # Label residues as binding (1) if annotated in BioLiP data
        # for residue_id in accumulator[key]:
        #    self.extract_labels(residue_id, accumulator[key], bs_set)

    def save_edge_features_to_csv(self, key, edge_features_df):
        edge_output_path = os.path.join(
            self.dirs["data"]["prop_templates"]["edge_properties"],
            f"{key}_edge_properties.csv.zip",
        )
        edge_features_df.to_csv(edge_output_path, index=False, compression="zip")

    def save_node_features_to_csv(self, key, residues, node_features_df):
        """
        Saves node and edge feature DataFrames to CSV files.

        Parameters:
            key (str): Identifier for the current PDB-chain combination.
            residues (dict): Dictionary of residue labels.
            node_features_df (pd.DataFrame): DataFrame with node features.
            edge_features_df (pd.DataFrame): DataFrame with edge features.
        """
        # pdb_id, chain_id = key.split("_")

        # Create labels DataFrame
        labels_df = pd.DataFrame(
            list(residues.items()), columns=["residue_id", "label"]
        )

        # Merge labels with node features
        node_combined_df = pd.merge(
            labels_df, node_features_df, on="residue_id", how="inner"
        )

        # Move label column to the end
        cols = [col for col in node_combined_df.columns if col != "label"] + ["label"]
        node_combined_df = node_combined_df[cols]

        # Save to CSV
        node_output_path = os.path.join(
            self.dirs["data"]["feat_templates"]["node_features"],
            f"{key}_node_features.csv.zip",
        )
        node_combined_df.to_csv(node_output_path, index=False, compression="zip")

        # print(f"[+] Saved node features to: {node_output_path}")
        # print(f"[+] Saved edge features to: {edge_output_path}")

    #############################################################
    # AUXILIARY FUNCTIONS
    #############################################################

    def check_protein_sequence(self, protein):
        """
        Check if a protein sequence is valid (not empty and not composed only of unknown residues).
        """
        return bool(protein.sequence) and not set(protein.sequence) == {"X"}

    def save_invalid_sequence_log(self, pdb_id, chain_id):
        """
        Log invalid protein sequences (empty or unknown-only) to a file.
        """
        with open("invalid_sequences.log", "a") as f:
            f.write(f"{pdb_id}_{chain_id}\n")

    def save_embedding_error_log(self, pdb_id, chain_id, error):
        """
        Log errors during embedding generation to a file.
        """
        with open("embedding_errors.log", "a") as f:
            f.write(f"{pdb_id}_{chain_id} - {error}\n")

    def extract_labels(self, residue_id, acc_key, bs_set):
        # Label residues as binding (1) if annotated in BioLiP data
        for residue_id in acc_key:
            resname, resnum, chain = residue_id.split("_")
            one_letter = three_to_one(resname)
            res_short = f"{one_letter}{resnum}"
            if res_short in bs_set:
                acc_key[residue_id] = 1

    def verify_pdb_file_existance(self, pdb_id, chain_id, pdb_path):
        # Check if PDB file exists; download if necessary
        if not os.path.isfile(pdb_path):
            print(f"[!] PDB {pdb_path} not found.")
            # download_pdb(pdb_id, self.pdb_dir)

        # Confirm the PDB file was successfully downloaded
        if not os.path.isfile(pdb_path):
            print(f"[x] Failed to download {pdb_id}. Skipping.")
            with open("missing_pdbs.txt", "a") as f:
                f.write(f"{pdb_id}_{chain_id}\n")
            return False

        return True
