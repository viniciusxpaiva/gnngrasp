from Bio.PDB import HSExposure, NeighborSearch, Selection, is_aa, NACCESS
from app.utils.utils import load_atom_types
import pandas as pd
import numpy as np
import os


class NodePropertiesExtractor:
    def __init__(self, protein, naccess_binary, naccess_out_dir):
        # Initialize the extractor with a path to a PDB file
        self.protein = protein
        self.atom_types_dict = {}
        self.naccess_binary = naccess_binary
        self.naccess_out_dir = naccess_out_dir

        # Load atom types into memory
        load_atom_types(self.atom_types_dict)

    def extract_node_properties(self):
        """
        Extracts and merges all node-level properties (atom properties, interactions, accessibility, exposure)
        into a single DataFrame, ensuring a consistent set of columns across all templates.

        If essential properties (atom or interactions) are missing, returns an empty DataFrame.

        Returns:
            pd.DataFrame: Final merged DataFrame containing all node properties.
        """

        # Define expected columns for each property group (excluding residue_id)
        expected_columns = {
            "node_atom_properties": [
                "aromatic",
                "acceptor",
                "donor",
                "hydrophobic",
                "positive",
                "negative",
            ],
            "node_interaction_properties": [
                "aromatic_bond",
                "disulfide_bridge",
                "hydrogen_bond",
                "hydrophobic_bond",
                "repulsive_bond",
                "salt_bridge",
            ],
            "node_accessibility_properties": [
                "acc_all",
                "acc_side",
                "acc_main",
                "acc_polar",
                "acc_apolar",
            ],
            "node_exposure_properties": ["hse_up", "hse_down"],
        }

        # Step 1: Extract individual property DataFrames
        node_atom_properties = self.get_atom_properties()
        node_interaction_properties = self.get_interaction_properties()
        node_accessibility_properties = self.get_solvent_accessibility_properties()
        node_exposure_properties = self.get_exposure_properties()

        # Step 2: Check essential properties (atom and interaction properties must exist)
        essential_dfs = [node_atom_properties, node_interaction_properties]
        if any(df is None or df.empty for df in essential_dfs):
            return pd.DataFrame()  # Return empty if crucial properties are missing

        # Step 3: Use node_atom_properties as the base DataFrame
        df_final = node_atom_properties.copy()

        # Step 4: Organize all extracted properties into a dictionary
        property_dfs = {
            "node_atom_properties": node_atom_properties,
            "node_interaction_properties": node_interaction_properties,
            "node_accessibility_properties": node_accessibility_properties,
            "node_exposure_properties": node_exposure_properties,
        }

        # Step 5: Merge other properties into the final DataFrame
        for attr, cols in expected_columns.items():
            if attr == "node_atom_properties":
                continue  # already used as base

            property_df = property_dfs.get(attr)
            if property_df is None or property_df.empty:
                # If missing, create a DataFrame filled with NaNs
                property_df = pd.DataFrame(
                    {
                        "residue_id": df_final["residue_id"],
                        **{col: [float("nan")] * len(df_final) for col in cols},
                    }
                )

            df_final = df_final.merge(property_df, on="residue_id", how="left")

        # Step 6: Ensure all expected columns are present
        for attr, cols in expected_columns.items():
            for col in cols:
                if col not in df_final.columns:
                    df_final[col] = float("nan")

        # Step 7: Reorder columns
        ordered_columns = ["residue_id"] + [
            col for cols in expected_columns.values() for col in cols
        ]
        df_final = df_final[ordered_columns]

        return df_final

    def get_atom_properties(self):
        # Extract atom-level properties for each residue
        residue_data = []

        # Loop over all residues in the structure
        for chain in self.protein.get_chains():
            for residue in chain:
                if residue.get_id()[0] != " " or not is_aa(
                    residue
                ):  # Skip heteroatoms and non-amino acid residues
                    continue

                res_name = residue.get_resname()
                res_num = residue.get_id()[1]

                # Count atom types (aromatic, donor, etc.) for this residue
                counts = self._count_atom_types(residue, res_name)
                # Save residue identifier and atom type counts
                residue_data.append(
                    {"residue_id": f"{res_name}_{res_num}_{chain.id}", **counts}
                )

        return pd.DataFrame(residue_data)

    def get_interaction_properties(self, distance_cutoff=6.0):
        # Calculate interaction properties for each residue
        residue_data = []

        # Flatten all atoms in the structure to use in NeighborSearch
        atoms = Selection.unfold_entities(self.protein.structure, "A")
        ns = NeighborSearch(atoms)

        for chain in self.protein.get_chains():
            # Loop over all residues to calculate interaction types

            for residue in chain:
                if residue.get_id()[0] != " " or not is_aa(residue):
                    continue

                res_name = residue.get_resname()
                res_num = residue.get_id()[1]
                residue_id = f"{res_name}_{res_num}_{self.protein.chain_id}"
                # Initialize interaction counts
                interactions = {
                    "aromatic_bond": 0,
                    "disulfide_bridge": 0,
                    "hydrogen_bond": 0,
                    "hydrophobic_bond": 0,
                    "repulsive_bond": 0,
                    "salt_bridge": 0,
                }

                residue_atoms = list(residue.get_atoms())
                for atom in residue_atoms:
                    # Search for neighboring atoms within cutoff distance
                    neighbors = ns.search(atom.coord, distance_cutoff, level="A")
                    for neighbor in neighbors:
                        neighbor_residue = neighbor.get_parent()
                        if neighbor_residue == residue:
                            continue  # Skip atoms from the same residue

                        distance = np.linalg.norm(atom.coord - neighbor.coord)
                        interaction_type = self._get_interaction_type(
                            atom, neighbor, distance
                        )
                        if interaction_type:
                            interactions[interaction_type] += 1

                residue_data.append({"residue_id": residue_id, **interactions})

        return pd.DataFrame(residue_data)

    def get_solvent_accessibility_properties(self):
        """
        Calculate solvent accessibility properties using NACCESS.
        Returns an empty DataFrame if NACCESS returns invalid results (-99.9).
        """
        base_tmp_dir = os.path.join(self.naccess_out_dir, f"tmp_{self.protein.pdb_id}")
        os.makedirs(base_tmp_dir, exist_ok=True)

        tmp_pdb_path = os.path.join(base_tmp_dir, f"{self.protein.pdb_id}.pdb")
        self.protein.save_single_model_pdb(tmp_pdb_path)

        # Run NACCESS
        _ = NACCESS.NACCESS(
            model=self.protein.structure,
            pdb_file=tmp_pdb_path,
            naccess_binary=self.naccess_binary,
            tmp_directory=base_tmp_dir,
        )

        # Locate internal NACCESS directory
        internal_tmp_dirs = [
            os.path.join(base_tmp_dir, d)
            for d in os.listdir(base_tmp_dir)
            if os.path.isdir(os.path.join(base_tmp_dir, d)) and d.startswith("tmp")
        ]

        if not internal_tmp_dirs:
            raise FileNotFoundError("No internal NACCESS tmp directory found.")

        internal_tmp_dirs.sort(key=os.path.getmtime, reverse=True)
        internal_dir = internal_tmp_dirs[0]

        rsa_files = [f for f in os.listdir(internal_dir) if f.endswith(".rsa")]
        if not rsa_files:
            raise FileNotFoundError(
                "No .rsa file found inside internal NACCESS tmp directory."
            )

        rsa_path = os.path.join(internal_dir, rsa_files[0])

        # Parse .rsa file and check for invalid values
        residue_data = []
        invalid_values = True  # Flag to check for invalid (-99.9) values

        with open(rsa_path) as f:
            for line in f:
                if not line.startswith("RES"):
                    continue
                resname = line[4:7].strip()
                chain = line[8].strip()
                resnum = line[9:13].strip()
                residue_id = f"{resname}_{resnum}_{chain}"

                acc_all = float(line[22:28].strip())
                acc_side = float(line[36:41].strip())
                acc_main = float(line[49:54].strip())
                acc_polar = float(line[75:89].strip())
                acc_apolar = float(line[62:67].strip())

                # Check if at least one valid value exists
                if acc_all != -99.9:
                    invalid_values = False

                residue_data.append(
                    {
                        "residue_id": residue_id,
                        "acc_all": acc_all,
                        "acc_side": acc_side,
                        "acc_main": acc_main,
                        "acc_polar": acc_polar,
                        "acc_apolar": acc_apolar,
                    }
                )

        # Return empty DataFrame if only invalid values found
        if invalid_values:
            return pd.DataFrame()

        return pd.DataFrame(residue_data)

    def get_exposure_properties(self):
        """
        Calculates Half Sphere Exposure (HSE) properties using HSExposureCB.
        If an error occurs (e.g., due to missing CB atoms), returns an empty DataFrame.
        """

        try:
            # May fail if some residues are malformed or missing CB atoms
            exp_cb = HSExposure.HSExposureCB(self.protein.structure[0])
        except Exception as e:
            with open("processing_errors.log", "a") as f:
                f.write(f"{self.protein.pdb_id} - {e}\n")
            return pd.DataFrame()

        residue_data = []

        for key in exp_cb.keys():
            res_chain = key[0]
            res_number = key[1][1]

            # Attempt to get residue, skip if residue not found
            try:
                residue = self.protein.structure[0][res_chain][res_number]
                res_name = residue.get_resname()
            except KeyError:
                continue

            residue_id = f"{res_name}_{res_number}_{res_chain}"

            hse_values = exp_cb[key]  # Direct access instead of using .get()

            residue_data.append(
                {
                    "residue_id": residue_id,
                    "hse_up": hse_values[0],
                    "hse_down": hse_values[1],
                }
            )

        return pd.DataFrame(residue_data)

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _count_atom_types(self, residue, res_name):
        # Count atom types for a given residue
        counts = {
            "aromatic": 0,
            "acceptor": 0,
            "donor": 0,
            "hydrophobic": 0,
            "positive": 0,
            "negative": 0,
        }

        # For each atom, increment counters if it belongs to a given type
        for atom in residue:
            atom_name = atom.get_name().strip()
            types = self.atom_types_dict.get((res_name, atom_name), [])

            if "ARM" in types or "AROMATIC" in types:
                counts["aromatic"] += 1
            if "ACP" in types or "ACCEPTOR" in types:
                counts["acceptor"] += 1
            if "DON" in types or "DONOR" in types:
                counts["donor"] += 1
            if "HPB" in types or "HYDROPHOBIC" in types:
                counts["hydrophobic"] += 1
            if "POS" in types or "POSITIVE" in types:
                counts["positive"] += 1
            if "NEG" in types or "NEGATIVE" in types:
                counts["negative"] += 1

        return counts

    def _get_interaction_type(self, atom1, atom2, distance):
        # Define interaction types between atoms based on given criteria

        # Get interaction type lists for each atom
        types1 = self.atom_types_dict.get(
            (atom1.get_parent().get_resname(), atom1.get_name().strip()), []
        )
        types2 = self.atom_types_dict.get(
            (atom2.get_parent().get_resname(), atom2.get_name().strip()), []
        )

        # Aromatic stacking: 2 aromatic atoms within 1.5–3.5 Å
        if "ARM" in types1 and "ARM" in types2 and 1.5 <= distance <= 3.5:
            return "aromatic_bond"

        # Hydrogen bond: 1 acceptor and 1 donor atom within 2.0–3.0 Å
        if (
            ("ACP" in types1 and "DON" in types2)
            or ("DON" in types1 and "ACP" in types2)
        ) and 2.0 <= distance <= 3.0:
            return "hydrogen_bond"

        # Hydrophobic: 2 hydrophobic atoms within 2.0–3.8 Å
        if "HPB" in types1 and "HPB" in types2 and 2.0 <= distance <= 3.8:
            return "hydrophobic_bond"

        # Repulsive: 2 atoms with same charge within 2.0–6.0 Å
        if (
            ("POS" in types1 and "POS" in types2)
            or ("NEG" in types1 and "NEG" in types2)
        ) and 2.0 <= distance <= 6.0:
            return "repulsive_bond"

        # Salt bridges: 2 atoms with opposite charges within 2.0–6.0 Å
        if (
            ("POS" in types1 and "NEG" in types2)
            or ("NEG" in types1 and "POS" in types2)
        ) and 2.0 <= distance <= 6.0:
            return "salt_bridge"

        # Disulfide bridge: 2 sulfur atoms (SG–SG from CYS) within 2.0–2.2 Å
        if atom1.element == "S" and atom2.element == "S" and 2.0 <= distance <= 2.2:
            return "disulfide_bridge"

        return None  # No interaction identified
