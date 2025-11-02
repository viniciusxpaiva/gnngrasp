import os
import argparse
from tqdm import tqdm
from colorama import Fore, Style, init

from app.pipeline import Pipeline  # ajuste conforme seu projeto

init(autoreset=True)  # colorama

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepBENDER pipeline")
    parser.add_argument(
        "-tpt",
        "--generate_templates",
        action="store_true",
        help="Generate template features and embeddings before prediction",
    )
    parser.add_argument(
        "-p",
        "--pdb",
        required=False,
        default="10mh.pdb",
        help="PDB file for prediction",
    )
    parser.add_argument(
        "-n",
        "--topn",
        type=int,
        default=1,
        help="Number of top similar templates to select",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    args = parser.parse_args()
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    pipeline = Pipeline(BASE_DIR)

    if args.generate_templates:
        print(Fore.CYAN + "[1] Extracting template features for nodes and edges")
        pipeline.templates_extractor.generate_template_features()

        print(Fore.CYAN + "[2] Generating global embeddings for templates")
        pipeline.templates_extractor.generate_global_embeddings_templates(
            pipeline.global_embeddings_generator,
            os.path.join(
                pipeline.dirs["templates"]["global_embeddings"],
                "global_embeddings_templates.csv",
            ),
        )

        print(Fore.CYAN + "[3] Generating residual embeddings for templates")

        template_files = os.listdir(
            os.path.join(pipeline.dirs["templates"]["node_features"])
        )

        for filename in tqdm(
            template_files,
            desc="Embedding templates",
            colour="green",
            leave=True,  # mantém a barra final após completar
        ):
            if not filename.endswith("_node.csv"):
                continue

            pdb_id = filename.replace("_node.csv", "")
            pdb_filename = f"{pdb_id.split('_')[0]}.pdb"
            pdb_path = os.path.join(pipeline.dirs["pdbs"], pdb_filename)
            chain_id = pdb_id.split("_")[1]

            pipeline.residue_embeddings_generator.save_embeddings_to_csv(
                protein_id=pdb_id,
                pdb_path=pdb_path,
                chain_id=chain_id,
                output_dir=pipeline.dirs["templates"]["residue_embeddings"],
            )

            tqdm.write(
                Fore.GREEN
                + f"[+] Embeddings for {pdb_id} saved to {pipeline.dirs['templates']['residue_embeddings']}"
            )

    print(Fore.YELLOW + f"[4] Making prediction for {args.pdb}")
    pipeline.predict_protein_with_embedding_selection(
        args.pdb, top_n=args.topn, epochs=args.epochs
    )
    print(Fore.GREEN + "✅ End of prediction")
