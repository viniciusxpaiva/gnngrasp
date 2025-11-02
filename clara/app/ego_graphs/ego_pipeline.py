import os
import pandas as pd
from app.protein import Protein
from app.properties.edge_properties_extractor import EdgePropertiesExtractor
from app.ego_graphs.residue_extractor.residue_neighbors_extractor import (
    ResidueNeighborsExtractor,
)
from app.ego_graphs.templates.ego_templates_extractor import EgoGraphTemplateExtractor


class EgoGraphPipeline:
    """
    Pipeline for generating ego-graph (local neighborhood graph) data per residue.
    Coordinates:
        - Protein structure loading
        - Edge interaction extraction (for neighbor detection)
        - Ego-graph neighborhood CSV extraction via ResidueNeighborsExtractor
    """

    def __init__(self, base_dir):
        """
        Initialize the pipeline: set paths, configs and output directories.

        Args:
            base_dir (str): Root directory for input/output.
            pipeline_params (dict): Parameters and settings for the pipeline.
        """
        self.base_dir = base_dir
        self.dirs = self._define_directories()

    def dynamic_prediction(
        self,
        prot_id,
        chain_id,
        input_protein_path,
        epochs,
    ):
        """
        Full Deep-GRaSP prediction pipeline using an ensemble-based interface.
        This includes both training and prediction steps.

        Pipeline steps:
            1. Load and validate the input protein
            2. Generate node/edge embeddings and features
            3. Select templates based on similarity
            4. Run Optuna to optimize GNN hyperparameters
            5. Initialize and train one or more GNN models
            6. Predict binding sites for the input protein
            7. Save predictions to a CSV file

        Args:
            prot_id (str): PDB ID of the input protein (e.g., "1abc").
            chain_id (str): Chain ID to be analyzed (e.g., "A").
            input_protein_path (str): Path to the input PDB file.
            epochs (int): Number of training epochs for each model.
        """

        # Step 1: Load and validate the input protein structure
        print("[!] Preparing input protein data")
        input_protein = Protein(prot_id, input_protein_path, chain_id)
        self._prepare_output_dir(prot_id, chain_id)
        input_protein.save_fasta(self.dirs["output"]["prot_out_dir"])

        if not input_protein.is_valid:
            raise ValueError(
                f"[ERROR] Input protein {prot_id} is invalid (empty sequence or structure)"
            )

        # Step 2: Generate node embeddings and handcrafted edge properties for input protein
        self._generate_input_embeddings(input_protein)
        self._generate_input_features(input_protein)

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _define_directories(self):
        """
        Define all relevant directories for the pipeline.

        Returns:
            dict: Structured directory paths.
        """
        dirs = {
            "base": self.base_dir,
            "data": {
                "base": os.path.join(self.base_dir, "data"),
                "pdbs": os.path.join(self.base_dir, "data", "pdbs"),
                "ego_templates": os.path.join(self.base_dir, "data", "ego_templates"),
                "biolip_dataset": os.path.join(
                    self.base_dir, "data", "test_biolip_binding_data.txt"
                ),
                "prop_templates": {
                    "base": os.path.join(self.base_dir, "data", "prop_templates"),
                    "node_properties": os.path.join(
                        self.base_dir, "data", "prop_templates", "node_properties"
                    ),
                    "edge_properties": os.path.join(
                        self.base_dir, "data", "prop_templates", "1edge_properties"
                    ),
                    # "edge_properties": "/home/vinicius/content-deep-bender/prop_templates/edge_properties",
                },
            },
        }
        return dirs

    def _prepare_output_dir(self, prot_id, chain_id=None):
        """
        Prepare a subdirectory inside the output folder for the current protein prediction.
        Updates self.dirs["output"]["current_output"].

        Args:
            prot_id (str): PDB ID of the input protein.
            chain_id (str): Chain ID of the input protein.
        """
        output_base = self.dirs["output"]["base"]
        output_subdir = os.path.join(output_base, f"{prot_id}_{chain_id}")
        os.makedirs(output_subdir, exist_ok=True)
        self.dirs["output"]["prot_out_dir"] = output_subdir
