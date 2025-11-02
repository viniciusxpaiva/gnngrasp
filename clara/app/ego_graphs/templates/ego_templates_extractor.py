import os
import pandas as pd


class EgoGraphTemplateExtractor:
    """
    Class to extract ego-graphs (neighbor CSVs) for all templates in the BioLiP dataset.
    """

    def __init__(self, dirs):
        """
        Args:
            dirs (dict): Dictionary containing directories and settings.
        """
        self.dirs = dirs

    def generate_ego_graph_templates(self):
        """
        Reads existing edge property CSVs and generates neighbor CSVs for each template.
        Each output CSV contains, for each source residue, its direct neighbors (targets).
        """
        print("[INFO] Generating ego-graph neighbor CSVs from existing edge files")
        count = 0

        # List all edge property files
        edge_files = os.listdir(self.dirs["data"]["distance_templates"]["edges"])

        for edge_file in edge_files:
            # Skip non-csv files
            if not edge_file.endswith(".csv") and not edge_file.endswith(".csv.zip"):
                continue

            # Extract template key (e.g., '1abc_A')
            if edge_file.endswith("_edge_distance.csv.zip"):
                key = edge_file.replace("_edge_distance.csv.zip", "")
            else:
                key = edge_file.replace("_edge_distance.csv", "")

            edge_path = os.path.join(
                self.dirs["data"]["distance_templates"]["edges"],
                edge_file,
            )

            output_path = os.path.join(
                self.dirs["data"]["distance_templates"]["neighbors6"],
                f"{key}_neighbors.csv.zip",
            )

            try:
                # Read edge CSV (handles zipped and unzipped)
                edge_df = pd.read_csv(
                    edge_path, compression="zip" if edge_file.endswith(".zip") else None
                )
                edge_df = edge_df[edge_df["distance"] <= 6.0]

                # Build neighbors dictionary: {source: [target1, target2, ...]}
                neighbors_df = (
                    edge_df.groupby("source")["target"]
                    .apply(lambda x: ",".join(map(str, x)))
                    .reset_index()
                )
                neighbors_df.rename(
                    columns={"source": "residue_id", "target": "neighbors"},
                    inplace=True,
                )

                neighbors_df["seqnum"] = neighbors_df["residue_id"].apply(
                    self.extract_seqnum
                )
                neighbors_df = neighbors_df.sort_values("seqnum").drop(columns="seqnum")
                neighbors_df.to_csv(output_path, index=False, compression="zip")
                count += 1
                if count % 1000 == 0:
                    print(f"[✓] Processed {count} ego-graph templates.")

            except Exception as e:
                print(f"[X] Failed for {edge_file}: {e}")

        print(
            f"[INFO] Ego-graph neighbor extraction finished: {count} templates processed."
        )

    def extract_seqnum(self, residue_id):
        """
        Extracts the residue number from a residue_id string like 'MET_1_A'.
        Returns an integer for proper sorting.
        """
        # Assumes format is always RESNAME_RESNUM_CHAIN
        parts = residue_id.split("_")
        if len(parts) == 3:
            try:
                return int(parts[1])
            except ValueError:
                return 0  # fallback if not an integer
        return 0
