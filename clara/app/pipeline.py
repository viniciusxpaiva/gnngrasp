import os
import torch
import numpy as np
import pandas as pd
from typing import Dict
from torch_geometric.data import Data
from app.utils.build_subgraphs_from_neighbors import (
    build_input_subgraphs_from_neighbors,
)
from app.utils.prediction_utils import cleanup_cuda

from app.protein import Protein

# from app.templates.embed_template_extractor import EmbedTemplateExtractor


class Pipeline:
    """
    Full pipeline to prepare templates, extract features, train the GNN model,
    and predict binding sites using ESM-based node embeddings and optional handcrafted features.

    This class coordinates:
    - Directory and file organization
    - Embedding and feature extraction
    - Configuration via prediction parameters
    """

    def __init__(self, base_dir, output_dir, prediction_params):
        """
        Initialize the pipeline by setting paths, parameters, and embedding generators.

        Args:
            base_dir (str): Base directory where all input/output folders are located.
            prediction_params (dict): Dictionary containing all configuration settings
                                      such as number of layers, GNN type, learning rate, etc.
        """
        self.base_dir = base_dir
        self.prediction_params = prediction_params
        self.use_node_properties = prediction_params["use_embd_projection"]
        self.dirs = self._define_directories(output_dir)
        #os.makedirs(self.dirs["output"]["base"], exist_ok=True)

        print(f"Saving resuls in: {output_dir}")

        # Initialize embedding generators
        # self.template_extractor = EmbedTemplateExtractor(self.dirs)

        # Templates
        # self.embd_templates_extractor = EmbedTemplateExtractor(self.dirs)

    def hier_prediction(self, prot_id, input_prot_path, params):

        from app.hier_prediction import HierPrediction

        # Step 1: Load and validate the input protein structure
        print("[!] Preparing input protein data")
        input_protein = Protein(prot_id, input_prot_path, "A")
        #self._prepare_output_dir(prot_id, "A")
        input_protein.save_fasta(self.dirs["output"]["prot_out_dir"])

        if not input_protein.is_valid:
            raise ValueError(
                f"[ERROR] Input protein {prot_id} is invalid (empty sequence or structure)"
            )

        self._generate_input_node_embeddings(input_protein)
        self._generate_input_edge_features(input_protein)

        input_subg_gen_method = params["subg_gen_method"]
        num_layers = params["subg_neighbor_layers"]
        if input_subg_gen_method == "color":
            input_subgraphs = self._build_input_subgraphs_color(
                input_protein, num_layers
            )
        elif input_subg_gen_method == "anchor":
            input_subgraphs = self._build_input_subgraphs_anchor(
                input_protein, num_layers
            )
        else:
            self._generate_input_node_properties(input_protein)
            exposure_percent = params["asa_exposure_percent"]
            input_subgraphs = self._build_input_subgraphs_asa(
                input_protein, num_layers, exposure_percent
            )

        self.prediction_pipeline = HierPrediction(self.dirs, params)
        self.prediction_pipeline.prepare_dataset(input_protein)
        print("AQUI")

        if params["use_subgraph_classifier"]:
            self.prediction_pipeline.initialize_subgraph_classifier()
            self.prediction_pipeline.train_subgraph_classifier()
            

        self.prediction_pipeline.initialize_node_classifier()
        self.prediction_pipeline.train_node_classifier()

        residue_scores = self.prediction_pipeline.predict_binding_sites(input_subgraphs)

        all_residues = sorted(set().union(*[g.ego_nodes for g in input_subgraphs]))
        
        print("Salvando predições")
        self._save_binding_site_predictions(
            residue_scores, prot_id, "A", all_residues
        )

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _define_directories(self, output_dir):
        """
        Define and organize all project directories.

        Returns:
            dict: Directory paths organized by purpose.
        """
        dirs = {
            "base": self.base_dir,
            "smote": {
                "base": os.path.join(self.base_dir, "smote"),
                "templates": os.path.join(self.base_dir, "smote", "templates"),
                "checkpoint": os.path.join(self.base_dir, "smote", "checkpoint"),
            },
            "app": {
                "base": os.path.join(self.base_dir, "app"),
                "gnn": os.path.join(self.base_dir, "app", "gnn"),
                "utils": os.path.join(self.base_dir, "app", "utils"),
            },
            "data": {
                "base": os.path.join(self.base_dir, "data"),
                "blast": os.path.join(self.base_dir, "data", "blast"),
                "atom_types": os.path.join(self.base_dir, "data", "atom_types.txt"),
                "smote_data": os.path.join(self.base_dir, "data", "smote_data"),
                "smote_input": os.path.join(self.base_dir, "data", "smote_input"),
                "smote_templates": os.path.join(
                    self.base_dir, "data", "smote_templates"
                ),
                "biolip_dataset": os.path.join(
                    self.base_dir, "data", "test_biolip_binding_data.txt"
                ),
                "ego_templates": os.path.join(self.base_dir, "data", "ego_templates"),
                "pdbs": os.path.join(self.base_dir, "data", "pdbs"),
                # "pdbs": "/home/vinicius/content-deep-bender/pdbs",
                # "pdbs": "/storage/hpc/11/dealmei1/content-deep-grasp/pdbs",
                "esm_templates": {
                    "base": os.path.join(self.base_dir, "data", "esm_templates"),
                    "node_embeddings": os.path.join(
                        self.base_dir, "data", "esm_templates", "node_embeddings"
                    ),
                    # "node_embeddings": "/home/vinicius/content-deep-bender/esm_templates/node_embeddings",
                    # "node_embeddings": "/storage/hpc/11/dealmei1/content-deep-grasp/esm_templates/node_embeddings",
                    "edge_embeddings": os.path.join(
                        self.base_dir, "data", "esm_templates", "edge_embeddings"
                    ),
                    "global_embeddings": os.path.join(
                        self.base_dir, "data", "esm_templates", "global_embeddings"
                    ),
                },
                "pbert_templates": {
                    "base": os.path.join(self.base_dir, "data", "pbert_templates"),
                    "node_embeddings": os.path.join(
                        self.base_dir, "data", "pbert_templates", "node_embeddings"
                    ),
                },
                "prop_templates": {
                    "base": os.path.join(self.base_dir, "data", "prop_templates"),
                    "node_properties": os.path.join(
                        self.base_dir, "data", "prop_templates", "node_properties"
                    ),
                    # "node_properties": "/home/vinicius/content-deep-bender/prop_templates/node_properties",
                    "edge_properties": os.path.join(
                        self.base_dir, "data", "prop_templates", "edge_properties"
                    ),
                    # "edge_properties": "/home/vinicius/content-deep-bender/prop_templates/edge_properties",
                    # "edge_properties": "/storage/hpc/11/dealmei1/content-deep-grasp/prop_templates/edge_properties",
                    "global_embeddings": os.path.join(
                        self.base_dir, "data", "prop_templates", "global_embeddings"
                    ),
                },
                "distance_templates": {
                    "base": os.path.join(self.base_dir, "data", "distance_templates"),
                    "edges": os.path.join(
                        self.base_dir, "data", "distance_templates", "edges"
                    ),
                    "neighbors": os.path.join(
                        self.base_dir, "data", "distance_templates", "neighbors"
                    ),
                },
            },
            "scripts": os.path.join(self.base_dir, "scripts"),
            "output": {
                "prot_out_dir": output_dir,
            },
            "naccess": {
                "binary": "/home/vinicius/naccess/naccess",
                "out_dir": os.path.join(self.base_dir, "data", "naccess_output"),
            },
        }
        return dirs

    def _build_input_subgraphs_anchor(
        self,
        protein,
        num_layers,
    ):
        """
        Generate subgraphs for the input protein, using every residue as an anchor (root).
        Each residue in the protein creates exactly one subgraph, built by expanding
        its k-hop neighborhood.

        This method ignores accessibility or redundancy heuristics, ensuring
        complete coverage.

        Args:
            protein (Protein): Protein object with node and edge features loaded in memory.
            num_layers (int): Number of neighbor expansion layers (k-hop expansion).

        Returns:
            List[Data]: List of PyG Data objects, where each Data corresponds to a
            subgraph rooted at a distinct anchor residue.
        """
        cutoff_residues = list(protein.node_embeddings["residue_id"])
        subgraphs = build_input_subgraphs_from_neighbors(
            protein,
            num_layers,
            cutoff_residues,
            subgraph_type="anchor",
            verbose=True,
        )
        return subgraphs

    def _build_input_subgraphs_asa(
        self,
        protein,
        num_layers,
        exposure_percent,
    ):
        """
        Generate subgraphs for the input protein using only solvent-exposed residues
        as anchors (roots). Each selected residue expands to a k-hop neighborhood.

        Unlike the anchor mode, only residues with solvent accessibility
        above `exposure_percent` are considered as roots.

        Args:
            protein (Protein): Protein object with node and edge features loaded in memory.
            num_layers (int): Number of neighbor expansion layers (k-hop expansion).
            exposure_percent (float): Accessibility threshold; residues with acc_all greater
                                    than this value are used as anchors.

        Returns:
            List[Data]: List of PyG Data objects, where each Data corresponds to a
            subgraph rooted at an exposed residue.
        """
        node_prop_df = protein.node_properties
        node_prop_df = node_prop_df[node_prop_df["acc_all"] > exposure_percent]
        cutoff_residues = list(node_prop_df["residue_id"])
        subgraphs = build_input_subgraphs_from_neighbors(
            protein,
            num_layers,
            cutoff_residues,
            subgraph_type="anchor",  # ASA behaves like anchor in coverage
            verbose=True,
        )
        return subgraphs

    def _build_input_subgraphs_color(
        self,
        protein,
        num_layers,
    ):
        """
        Generate subgraphs for the input protein using a coloring/coverage scheme.
        All residues are potential anchors, but coverage is updated with all nodes
        included in each subgraph, reducing redundancy.

        This ensures that the set of subgraphs collectively covers all residues
        with fewer overlapping neighborhoods.

        Args:
            protein (Protein): Protein object with node and edge features loaded in memory.
            num_layers (int): Number of neighbor expansion layers (k-hop expansion).

        Returns:
            List[Data]: List of PyG Data objects, covering the entire protein with
            minimal redundancy.
        """
        cutoff_residues = list(protein.node_embeddings["residue_id"])
        subgraphs = build_input_subgraphs_from_neighbors(
            protein,
            num_layers,
            cutoff_residues,
            subgraph_type="color",
            verbose=True,
        )
        return subgraphs

    def _generate_input_node_embeddings(self, protein, save_csv=False):
        """
        Generate node-level embeddings for the input protein.
        Handles duplicates and missing values for downstream compatibility.

        Args:
            protein (Protein): Protein object.
            save_csv (bool): If True, save the extracted embeddings to CSV.
        """
        # Generate embeddings

        input_csv_file = f"{self.dirs["data"]["esm_templates"]["node_embeddings"]}/{protein.pdb_id}_node_embeddings.csv"
        if os.path.exists(input_csv_file):
            protein.node_embeddings = pd.read_csv(input_csv_file)
            print("[✓] Using existing CSV file for input protein node embeddings")
        else:
            print("[✓] Extracting node embeddings for input protein")
            from app.embeddings.esm_node_embeddings_generator import (
                ESMNodeEmbeddingsGenerator,
            )

            self.node_embeddings_generator = ESMNodeEmbeddingsGenerator()
            df = self.node_embeddings_generator.generate_node_embeddings(protein)
            # cleanup_cuda(self.node_embeddings_generator.model)

            # Clean the DataFrame
            df = df.drop_duplicates(
                subset="residue_id", keep="first"
            )  # Remove duplicates
            df = df.fillna(0)  # Replace NaNs with 0

            # Assign back to protein object
            protein.node_embeddings = df

            df.to_csv(input_csv_file, index=False)

            # Optionally save to CSV
            if save_csv:
                output_filename = (
                    f"{protein.pdb_id}{protein.chain_id}_node_embeddings.csv"
                )
                output_path = os.path.join(
                    self.dirs["output"]["prot_out_dir"], output_filename
                )
                df.to_csv(output_path, index=False)

                print(f"[✓] Saved cleaned node embeddings to {output_path}")

    def _generate_input_node_properties(self, protein, save_csv=False):
        """
        Generate all types of node-level properties for the input protein.
        Handles duplicates and missing values for downstream compatibility.

        Args:
            protein (Protein): Protein object.
            save_csv (bool): If True, save the extracted properties to CSV.
        """

        input_csv_file = f"{self.dirs["data"]["prop_templates"]["node_properties"]}/{protein.pdb_id}_node_properties.csv"
        if os.path.exists(input_csv_file):
            protein.node_properties = pd.read_csv(input_csv_file)
            print("[✓] Using existing CSV file for input protein node properties ")

        else:
            print("[✓] Extracting node properties for input protein")
            # Initialize extractor
            from app.properties.node_properties_extractor import NodePropertiesExtractor

            self.node_properties_extractor = NodePropertiesExtractor(
                protein,
                self.dirs["naccess"]["binary"],
                self.dirs["output"]["prot_out_dir"],
            )

            # Extract properties
            df = self.node_properties_extractor.extract_node_properties()

            # Clean the DataFrame
            df = df.drop_duplicates(
                subset="residue_id", keep="first"
            )  # Remove duplicates
            df = df.fillna(0)  # Replace NaNs with 0

            # Assign back to protein object
            protein.node_properties = df
            df.to_csv(input_csv_file, index=False)

            # Optionally save to CSV
            if save_csv:
                output_filename = (
                    f"{protein.pdb_id}{protein.chain_id}_node_properties.csv"
                )
                output_path = os.path.join(
                    self.dirs["output"]["prot_out_dir"], output_filename
                )
                df.to_csv(output_path, index=False)
                print(f"[✓] Saved cleaned node properties to {output_path}")

    def _generate_input_edge_features(self, protein):
        """
        Extract node-level graph features (node properties) for the input protein
        using the NodePropertiesExtractor.

        This function:
        - Initializes the NodePropertiesExtractor if not already done.
        - Runs the feature extraction.
        - Assigns the resulting node properties to the Protein object.

        Args:
            protein (Protein): The input protein object to process.
        """

        input_csv_file = f"{self.dirs["data"]["prop_templates"]["edge_properties"]}/{protein.pdb_id}_edge_properties.csv"
        if os.path.exists(input_csv_file):
            protein.edge_properties = pd.read_csv(input_csv_file)
            print("[✓] Using existing CSV file for input protein edge properties ")

        else:
            print("[✓] Extracting edge features for input protein")
            from app.properties.edge_properties_extractor import EdgePropertiesExtractor

            self.edge_properties_extractor = EdgePropertiesExtractor(protein)
            df = self.edge_properties_extractor.get_edge_features()
            protein.edge_properties = df
            df.to_csv(input_csv_file, index=False)

    def _save_binding_site_predictions(
        self,
        residue_scores: Dict[str, float],
        prot_id: str,
        chain_id: str,
        all_residue_ids: list[str],
        threshold: float = 0.5,
    ):
        """
        Process and save residue-level binding site predictions to a CSV file.

        Args:
            residue_scores (dict): Mapping of residue_id to predicted binding site score.
            prot_id (str): PDB ID of the input protein.
            chain_id (str): Chain ID of the input protein.
            all_residue_ids (list): Complete list of residues from the input protein.
            threshold (float): Threshold to convert scores into binary labels (default: 0.5).
        """
        # === Step 1: Ensure all residues are present ===
        complete_scores = {
            res_id: residue_scores.get(res_id, 0.0) for res_id in all_residue_ids
        }

        predictions_df = pd.DataFrame(
            [
                {
                    "residue_id": residue_id,
                    "binding_score": round(score, 2),
                    "predicted_label": int(score >= threshold),
                }
                for residue_id, score in complete_scores.items()
            ]
        )

        predictions_df["residue_number"] = (
            predictions_df["residue_id"].str.extract(r"_([0-9]+)_").astype(int)
        )
        predictions_df = predictions_df.sort_values(by="residue_number").drop(
            columns="residue_number"
        )

        # === Step 2: Save ===
        output_path = self._save_prediction_to_csv(predictions_df, prot_id, chain_id)
        print(f"[✓] Predictions saved to: {output_path}")

    def _save_prediction_to_csv(self, predictions_df, prot_id, chain_id):
        """
        Save the prediction results to a standardized CSV file.

        Args:
            predictions_df (DataFrame): DataFrame containing the predicted labels.
            prot_id (str): PDB ID of the input protein.
            chain_id (str): Chain ID of the input protein.

        Returns:
            str: Path where the file was saved.
        """

        # Compose output filename

        output_filename = f"{prot_id}_prediction.csv"
        #output_path = os.path.join(self.dirs["output"]["prot_out_dir"], output_filename)
        
        output_path = self.dirs["output"]["prot_out_dir"] + "/" + output_filename
        # Save the DataFrame
        predictions_df.to_csv(output_path, index=False)
        return output_path

    def _prepare_output_dir(self, prot_id, chain_id=None):
        """
        Prepare a subdirectory inside the output folder for the current protein prediction.
        Updates self.dirs["output"]["current_output"].

        Args:
            prot_id (str): PDB ID of the input protein.
            chain_id (str): Chain ID of the input protein.
        """
        output_base = self.dirs["output"]["base"]
        #output_subdir = os.path.join(output_base, f"{prot_id}_{chain_id}")
        #os.makedirs(output_subdir, exist_ok=True)
        self.dirs["output"]["prot_out_dir"] = output_base
