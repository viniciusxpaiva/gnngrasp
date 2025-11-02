import os
import torch
import numpy as np
import pandas as pd
from typing import Iterable, List
from torch_geometric.data import Data


def build_input_subgraphs_from_neighbors(
    protein,
    num_layers: int,
    cutoff_residues: Iterable[str],
    subgraph_type: str,
    verbose: bool = True,
) -> List[Data]:
    """
    Build k-hop ego-subgraphs for an input protein using its adjacency information.

    Each subgraph is rooted at one residue (anchor) from `cutoff_residues` and
    expanded with a BFS up to `num_layers` hops. Node embeddings are taken from
    `protein.node_embeddings`, and graph edges from `protein.edge_properties`.

    Coverage policy (controlled by `subgraph_type`):
      - "color": after creating a subgraph, mark ALL nodes in it as covered,
                 reducing redundancy across subgraphs.
      - otherwise ("anchor"/"asa"): mark ONLY the root residue as covered,
                 ensuring one subgraph per root residue.

    Args:
        protein: Protein object with attributes:
            - node_embeddings (DataFrame): rows = residues, includes "residue_id" + features
            - edge_properties (DataFrame): rows = edges, includes "source" and "target"
        num_layers: Number of k-hop neighbor expansions (BFS depth).
        cutoff_residues: Candidate residues to be used as subgraph roots.
        subgraph_type: "color", "anchor", or "asa" (defines coverage policy).
        verbose: If True, print debug information.

    Returns:
        List[Data]: One PyG Data object per constructed subgraph, with fields:
            - x: (N, F) node features
            - edge_index: (2, E) bidirectional edge list
            - edge_attr: None (placeholder for future edge features)
            - ego_center: root residue
            - ego_nodes: list of residue_ids in the subgraph
            - input_id: protein identifier (pdb_id + chain_id)
    """

    # --- Load node embeddings and edge properties ---
    node_df = protein.node_embeddings
    edge_df = protein.edge_properties

    # Build adjacency mapping {node -> [neighbors]}
    neighbors_dict = {}
    for _, row in edge_df.iterrows():
        src, tgt = row["source"], row["target"]
        neighbors_dict.setdefault(src, []).append(tgt)

    # Map residue_id -> feature vector (embedding)
    residue_map = {
        row.residue_id: row.drop(["residue_id"]).astype(float).values
        for _, row in node_df.iterrows()
    }

    subgraphs: List[Data] = []
    covered_residues: set[str] = set()

    # --- Iterate over candidate root residues ---
    for center_res in cutoff_residues:
        if center_res in covered_residues:
            continue

        # BFS expansion up to num_layers
        visited = {center_res}
        frontier = {center_res}
        for _ in range(max(0, num_layers)):
            next_frontier = set()
            for node in frontier:
                for nbr in neighbors_dict.get(node, []):
                    if nbr in residue_map and nbr not in visited:
                        next_frontier.add(nbr)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier

        # Anchor first, then neighbors sorted for determinism
        sub_nodes = [center_res] + sorted(n for n in visited if n != center_res)
        node_to_idx = {n: i for i, n in enumerate(sub_nodes)}

        # --- Build bidirectional edge_index ---
        edge_pairs = []
        for src in sub_nodes:
            for tgt in neighbors_dict.get(src, []):
                if src in node_to_idx and tgt in node_to_idx:
                    i, j = node_to_idx[src], node_to_idx[tgt]
                    if i <= j:  # avoid duplicates
                        edge_pairs.append([i, j])
                        edge_pairs.append([j, i])

        edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long).t()
            if edge_pairs
            else torch.empty((2, 0), dtype=torch.long)
        )

        # --- Node feature matrix ---
        x = torch.from_numpy(np.asarray([residue_map[n] for n in sub_nodes])).float()

        # Get index of the center residue in the subgraph
        center_local = node_to_idx[center_res]

        # --- Assemble PyG Data object ---
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=None,  # placeholder (no edge features used here)
            ego_center=center_res,
            ego_center_index=torch.tensor(center_local, dtype=torch.long),
            ego_nodes=sub_nodes,
            input_id=f"{protein.pdb_id}_{protein.chain_id}",
        )

        subgraphs.append(data)

        # --- Coverage policy ---
        if subgraph_type == "color":
            covered_residues.update(sub_nodes)
        else:  # "anchor" / "asa"
            covered_residues.add(center_res)

    if verbose:
        print(
            f"[✓] Built {len(subgraphs)} subgraphs ({num_layers}-layer neighbors) "
            f"for input protein with {len(cutoff_residues)} residues."
        )

    return subgraphs


def build_template_subgraphs_from_neighbors(
    template_id: str,
    node_embd_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    cutoff_residues: Iterable[str],
    num_layers: int,
    subgraph_type: str,
) -> List[Data]:
    """
    Build k-hop ego-subgraphs for a template protein using a precomputed neighbor table.

    This utility constructs PyTorch Geometric Data objects from:
      - node_embd_df: per-residue embeddings + binary label (binding-site or not)
      - neighbors_df: per-residue neighbor lists (as comma-separated strings)
      - cutoff_residues: candidate residues to serve as subgraph anchors (roots)

    Coverage policy (controlled by `subgraph_type`):
      - "color": after creating a subgraph, mark ALL subgraph nodes as covered
                 (reduces redundancy across subgraphs).
      - otherwise (e.g., "anchor"/"asa"): mark ONLY the root as covered
                 (one subgraph per root, regardless of overlap).

    Args:
        template_id: Identifier of the template (e.g., "1abc_A").
        node_embd_df: DataFrame with columns ["residue_id", <embedding cols...>, "label"].
                      Embedding cols are numeric; "label" is 0/1 per residue.
        neighbors_df: DataFrame with columns ["residue_id", "neighbors"] where "neighbors"
                      is a comma-separated string of residue_ids (or NaN).
        cutoff_residues: Iterable of residue_ids to be used as subgraph roots.
        num_layers: Number of k-hop expansions from the root (BFS layers).
        subgraph_type: "color" or another mode ("anchor"/"asa") that affects coverage.

    Returns:
        List[Data]: One Data per constructed subgraph. Subgraphs with no edges are skipped.
                    Each Data contains:
                      - x: (N, F) node features
                      - edge_index: (2, E) directed (bidirectional) edges
                      - y: (1,) graph label (1 if any node is binding-site)
                      - site_ratio: scalar tensor = (#positives / N)
                      - ego_center, ego_nodes, node_labels, template_id
    """

    # --- Validate residue universe from embeddings table ---
    valid_residues = set(node_embd_df["residue_id"].astype(str))

    # --- Clean and normalize neighbors table to keep only valid residues ---
    def _filter_neighbors(neighbor_str: str) -> str:
        """Drop neighbors not present in `valid_residues` and normalize NaNs to empty."""
        if pd.isna(neighbor_str) or neighbor_str == "":
            return ""
        keep = [n for n in neighbor_str.split(",") if n in valid_residues]
        return ",".join(keep)

    neighbors_df = neighbors_df.copy()
    neighbors_df["residue_id"] = neighbors_df["residue_id"].astype(str)
    neighbors_df = neighbors_df[neighbors_df["residue_id"].isin(valid_residues)]
    neighbors_df["neighbors"] = neighbors_df["neighbors"].apply(_filter_neighbors)

    # Build adjacency dict: residue_id -> list of neighbor residue_ids
    neighbors_dict = {
        rid: (nbrs.split(",") if isinstance(nbrs, str) and nbrs else [])
        for rid, nbrs in zip(neighbors_df["residue_id"], neighbors_df["neighbors"])
    }

    # --- Build node feature/label map for fast lookup ---
    # Expect columns: ["residue_id", *embedding_columns..., "label"]
    residue_map = {
        str(row.residue_id): {
            "embedding": row.drop(labels=["residue_id", "label"]).astype(float).values,
            "label": int(row.label),
        }
        for _, row in node_embd_df.iterrows()
    }

    # Pre-filter cutoff_residues to those present in the residue/neighbor space
    cutoff_residues = [str(r) for r in cutoff_residues if str(r) in valid_residues]

    subgraphs: List[Data] = []
    covered_residues: set[str] = set()

    for center_res in cutoff_residues:
        # Skip roots already covered (depending on the chosen policy)
        if center_res in covered_residues:
            continue

        # --- k-hop BFS expansion from the root ---
        visited = {center_res}
        frontier = {center_res}
        for _ in range(max(0, num_layers)):
            next_frontier = set()
            for node in frontier:
                for nbr in neighbors_dict.get(node, []):
                    if nbr in residue_map and nbr not in visited:
                        next_frontier.add(nbr)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier

        # Anchor first, then neighbors sorted for determinism
        sub_nodes = [center_res] + sorted(n for n in visited if n != center_res)

        # --- Build bidirectional edge_index (i<=j to avoid duplicate undirected pairs) ---
        node_to_idx = {n: i for i, n in enumerate(sub_nodes)}
        edge_pairs = []
        for src in sub_nodes:
            for tgt in neighbors_dict.get(src, []):
                if src in node_to_idx and tgt in node_to_idx:
                    i, j = node_to_idx[src], node_to_idx[tgt]
                    if i <= j:
                        edge_pairs.append([i, j])
                        edge_pairs.append([j, i])

        edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long).t()
            if edge_pairs
            else torch.empty((2, 0), dtype=torch.long)
        )

        # Skip degenerate subgraphs with no edges (isolated nodes)
        if edge_index.numel() == 0:
            continue

        # --- Assemble node feature matrix (x) ---
        x = torch.from_numpy(
            np.asarray([residue_map[n]["embedding"] for n in sub_nodes], dtype=float)
        ).float()

        # --- Node-level labels and graph-level label/ratio ---
        node_labels = [residue_map[n]["label"] for n in sub_nodes]
        subgraph_label = int(
            any(node_labels)
        )  # graph label = 1 if any node is positive
        site_ratio = torch.tensor(
            sum(node_labels) / len(node_labels), dtype=torch.float
        )

        # Get index of the center residue in the subgraph
        center_local = node_to_idx[center_res]

        # --- Build PyG Data object ---
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=None,  # add if you later include edge features
            y=torch.tensor([subgraph_label], dtype=torch.long),
            site_ratio=site_ratio,
            ego_center=center_res,
            ego_center_index=torch.tensor(center_local, dtype=torch.long),
            template_id=template_id,
            ego_nodes=sub_nodes,
            node_labels=torch.tensor(node_labels, dtype=torch.long),  # for step 2
        )
        subgraphs.append(data)

        # --- Coverage update policy ---
        if subgraph_type == "color":
            # Mark the entire subgraph as covered to reduce redundancy
            covered_residues.update(sub_nodes)
        else:
            # Only the root is marked as covered (anchor/asa-style behavior)
            covered_residues.add(center_res)
    return subgraphs
