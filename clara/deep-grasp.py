import os
import time
import copy
import numpy as np
import argparse

# from app.templates.distance_edge_generator import EdgeDistanceGenerator
# from app.embeddings.esm_node_embeddings_generator import ESMNodeEmbeddingsGenerator
from app.pipeline import Pipeline
#from app.utils.experiments_utils import run_planetoid_grid_experiments

# from app.ego_graphs.templates.ego_templates_extractor import EgoGraphTemplateExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepBENDER pipeline")
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory for predictions",
    )
    args = parser.parse_args()

    print("Starting Deep-GRaSP")

    DPS_TEST = False

    # TEST_TYPE = "coach/full_coach/"
    #TEST_TYPE = "coach/coach100/"
    # TEST_TYPE = "coach/teste/"
    # TEST_TYPE = "coach/tmp2/"
    TEST_TYPE = "../input/"

    INPUT_PDB_DIR = f"/home/vinicius/Desktop/deep-grasp/experiments/{TEST_TYPE}"
    DPS_TEST_FILE = "experiments/deepprosite/tmp_test.fa"

    PREDICTION_PARAMS = {
        "top_n_templates": 30,
        "sub_model": False,
        "num_gnn_layers": 2,
        "use_embd_projection": False,
        "use_edge_attr": False,
        "baseline": False,
        "hier": True,
        "smote": False,
        "node_import": False,
        "bat": False,
        "epochs": 100,
        "embd_type": "ESM",
        "layer_type": "GAT",
        "gnn_params": {
            "hidden_dim": 128,
            "lr": 0.001,
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "num_heads": 6,
            "projected_embd_dim": 128,
        },
        "cnn_params": {
            "cnn_channels": 32,
            "cnn_kernel_size": 5,
            "cnn_dropout": 0.2,
        },
        "mlp_params": {
            "mlp_hidden": 32,
            "mlp_dropout": 0.2,
        },
        "prediction_threshold": 0.5,
    }

    # Configuration dictionary for hierarchical subgraph classification
    HIER_PARAMS = {
        "subg_gen_method": "color",  # asa | color | anchor
        "subg_neighbor_layers": 3,
        "use_subgraph_classifier": True,
        "use_subgraph_filtering": True,
        "multiclass_decision_rule": "logit_margin",  # "argmax" | "margin" | "logit_margin"
        # "all_or_pos_subg_node_training": "all",  # all | pos
        "top_n_templates": 30,
        "asa_exposure_percent": 60,
        "subgraph_classifier": {
            "gnn_type": "GAT",  # GCN | GAT | GIN | SAGE | PNA
            "norm_type": "batch",  #
            "num_layers": 2,
            "pool_type": "add",
            "hidden_dim": 128,
            "output_dim": 1,
            "num_heads": 4,
            "dropout": 0.2,
            "lr": 1e-4,
            "epochs": 150,
            "weight_decay": 1e-5,
            "prediction_threshold": 0.5,
            "metric_mode": "f_beta",
            "beta": 1,
        },
        "node_classifier": {
            "gnn_type": "GAT",  # GCN | GAT | GIN | SAGE
            "norm_type": "batch",
            "num_layers": 1,
            "hidden_dim": 128,
            "num_heads": 4,
            "dropout": 0.2,
            "lr": 1e-4,
            "epochs": 60,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "fusion_mode": "none",  # concat | film | none
            "context_norm": "layernorm",  # 'none' | 'layernorm' | 'batch_zscore' | 'l2'
            "context_anneal": "none",  # 'none'|'linear'|'power'
        },
    }

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    CREATE_TEMPLATES = False
    COACH_TEST = True

    if PREDICTION_PARAMS["baseline"]:
        # --- grade de parâmetros ---
        GNN1_TYPES = ["GCN"]
        GNN1_LAYERS = [1]

        SG_NEIGHBORS = [5]

        GNN2_TYPES = ["GCN"]
        GNN2_LAYERS = [2]

        POOL_TYPE = ["add"]

        n_base_runs = 10
        dataset_name = "PubMed"

        pipeline = Pipeline(BASE_DIR, args.output_dir, PREDICTION_PARAMS)
        run_planetoid_grid_experiments(
            pipeline,
            HIER_PARAMS,
            n_base_runs,
            dataset_name,
            GNN1_TYPES,
            GNN1_LAYERS,
            SG_NEIGHBORS,
            GNN2_TYPES,
            GNN2_LAYERS,
            POOL_TYPE,
        )

    if CREATE_TEMPLATES:
        print(f"[0] Generating node, edge and global embeddings for templates")
        print(
            f"[0] Generating node and edge features and also global embeddings for templates"
        )
        dirs = {
            "data": {
                "distance_templates": {
                    "base": os.path.join(BASE_DIR, "data", "distance_templates"),
                    "edges": os.path.join(
                        BASE_DIR, "data", "distance_templates", "edges"
                    ),
                    "neighbors6": os.path.join(
                        BASE_DIR, "data", "distance_templates", "neighbors6"
                    ),
                },
            },
        }
        extractor = EgoGraphTemplateExtractor(dirs)
        extractor.generate_ego_graph_templates()

    if COACH_TEST:
        pipeline = Pipeline(BASE_DIR, args.output_dir, PREDICTION_PARAMS)
        # --- Step 1: List all PDB files in the folder ---

        pdb_files = [f for f in os.listdir(INPUT_PDB_DIR) if f.endswith(".pdb")]

        print(f"[+] Found {len(pdb_files)} PDB files to predict.")
        total_start_time = time.time()
        per_protein_times = []

        # --- Step 2: Run prediction for each file ---
        cont = 0
        for pdb_file in pdb_files:
            cont += 1
            filename = os.path.splitext(pdb_file)[0]  # remove ".pdb"

            if TEST_TYPE.startswith("casp10"):
                prot_id = filename
                chain_id = ""
            else:
                prot_id = filename[:-1]  # all except last character
                chain_id = filename[-1]  # last character

            input_protein_path = os.path.join(INPUT_PDB_DIR, pdb_file)

            csv_path = f"{args.output_dir}_output/{prot_id}_{chain_id}/{prot_id}{chain_id}_prediction.csv"

            if os.path.exists(csv_path):
                pass
                #continue

            print("--------------------------------------")
            print(
                f"Starting prediction for protein {prot_id}{chain_id} | ({cont}|{len(pdb_files)})"
            )
            print("--------------------------------------")

            protein_start_time = time.time()

            if PREDICTION_PARAMS["hier"]:
                print(
                    f"*** HIER PREDICTION | {HIER_PARAMS["subg_gen_method"].upper()} {HIER_PARAMS["asa_exposure_percent"]}% Subgraph ***"
                )
                pipeline.hier_prediction(
                    prot_id,
                    chain_id,
                    input_protein_path,
                    HIER_PARAMS,
                )
            elif PREDICTION_PARAMS["sub_model"]:
                print(
                    f"*** {PREDICTION_PARAMS["embd_type"]} | SUB-MODEL EMBEDDINGS PREDICTION ***"
                )
                pipeline.prediction(
                    prot_id,
                    chain_id,
                    input_protein_path,
                )
            elif PREDICTION_PARAMS["smote"]:
                print(f"*** {PREDICTION_PARAMS["embd_type"]} | SMOTE PREDICTION ***")
                pipeline.smote_prediction(
                    prot_id,
                    chain_id,
                    input_protein_path,
                )
            elif PREDICTION_PARAMS["bat"]:
                print(f"*** {PREDICTION_PARAMS["embd_type"]} | BAT PREDICTION ***")
                pipeline.bat_prediction(
                    prot_id,
                    chain_id,
                    input_protein_path,
                )
            elif PREDICTION_PARAMS["node_import"]:
                print(
                    f"*** {PREDICTION_PARAMS["embd_type"]} | NODE IMPORT PREDICTION ***"
                )
                pipeline.nodeimport_prediction(
                    prot_id,
                    chain_id,
                    input_protein_path,
                )
            else:
                print(
                    f"*** {PREDICTION_PARAMS["embd_type"]} | GNN-ONLY EMBEDDINGS PREDICTION ***"
                )
                pipeline.prediction_gnn(
                    prot_id,
                    chain_id,
                    input_protein_path,
                )

            protein_elapsed = time.time() - protein_start_time
            per_protein_times.append(protein_elapsed)

            print(
                f"[✓] Finished prediction for {prot_id}{chain_id} ({protein_elapsed:.2f} sec)\n"
            )
        total_elapsed = time.time() - total_start_time
        minutes = total_elapsed // 60
        seconds = total_elapsed % 60

        print(
            f"[✓] All predictions completed. Predictions saved at: {args.output_dir}_output"
        )
        print(f"[⏱️] Total prediction time: {int(minutes)} min {int(seconds)} sec.")

        if per_protein_times:
            avg_time = sum(per_protein_times) / len(per_protein_times)
            print(f"[⏱️] Average time per prediction: {avg_time:.2f} sec.")
