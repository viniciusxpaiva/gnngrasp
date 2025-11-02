import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(input_vector, template_vectors, method="cosine"):
    """
    Calculate similarity between an input vector and a set of template vectors.

    Args:
        input_vector (np.ndarray): 1D array for the input protein.
        template_vectors (np.ndarray): 2D array for all templates.
        method (str): Method to calculate similarity:
            - "cosine"
            - "dot_product"
            - "euclidean" (negative distance to keep similarity interpretation)

    Returns:
        np.ndarray: Array of similarity scores (one per template).
    """
    if method == "cosine":
        return cosine_similarity(
            input_vector.reshape(1, -1), template_vectors
        ).flatten()

    elif method == "dot_product":
        return np.dot(template_vectors, input_vector)

    elif method == "euclidean":
        distances = np.linalg.norm(template_vectors - input_vector, axis=1)
        return -distances  # Negative so higher = more similar

    else:
        raise ValueError(f"[ERROR] Unknown similarity method: {method}")


def prepare_node_tensor(node_df):
    """
    Convert node features DataFrame into a torch tensor (excluding residue ID).

    Args:
        node_df (pd.DataFrame): Node features DataFrame.

    Returns:
        torch.FloatTensor: Node feature tensor.
    """
    return torch.tensor(
        node_df.drop(["residue_id"], axis=1).values,
        dtype=torch.float,
    )


def prepare_edge_data(edge_df, node_df):
    """
    Convert edge information into PyG-compatible tensors.

    Args:
        edge_df (pd.DataFrame): Edge features DataFrame.
        node_df (pd.DataFrame): Node features DataFrame.

    Returns:
        edge_index (torch.LongTensor): Edge indices.
        edge_features (torch.FloatTensor): Edge attributes.
    """
    residue_to_index = {rid: i for i, rid in enumerate(node_df["residue_id"])}

    edge_indices = []
    valid_edge_features = []

    edge_features_cols = edge_df.columns.difference(["source", "target"])

    for idx, (src, tgt) in edge_df[["source", "target"]].iterrows():
        if src in residue_to_index and tgt in residue_to_index:
            values = edge_df.loc[idx, edge_features_cols].values.astype(float)
            edge_indices.append([residue_to_index[src], residue_to_index[tgt]])
            valid_edge_features.append(values)
        else:
            print(f"[WARNING] Residues not found: {(src, tgt)}")

    edge_index_input = torch.tensor(edge_indices, dtype=torch.long).t()
    edge_features_input = torch.from_numpy(np.vstack(valid_edge_features)).float()

    return edge_index_input, edge_features_input


def get_info_pos_neg_data(data_list):
    """
    Print a summary of the class balance (binding vs non-binding residues)
    across the loaded training dataset.

    Args:
        data_list (List[Data]): List of PyG Data objects, each representing a protein graph.

    Returns:
        Tuple[int, int]: Number of binding site residues (positives) and non-binding residues (negatives).
    """

    # Concatenate all labels across all proteins
    all_labels = torch.cat([data.y for data in data_list])

    # Calculate number of positives (binding sites) and negatives (non-binding sites)
    num_positive = (all_labels == 1).sum().item()
    num_negative = (all_labels == 0).sum().item()
    total = num_positive + num_negative

    # Calculate class proportions
    positive_ratio = num_positive / total
    negative_ratio = num_negative / total

    # Print data balance summary
    print("[DATA BALANCE SUMMARY]")
    print(f"Total residues: {total}")
    print(
        f"Binding site residues (class 1): {num_positive} ({positive_ratio:.2%}) | ",
        end="",
    )
    print(f"Non-binding site residues (class 0): {num_negative} ({negative_ratio:.2%})")

    return num_positive, num_negative


def plot_learning_curve(losses, split_idx=None, save_path=None):
    """
    Plot the training loss curve.

    Args:
        losses (List[float]): List of average loss per epoch.
        split_idx (int, optional): Index of the training split (for ensembles).
        save_path (str, optional): Path to save the plot as an image.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title = f"Learning Curve"
    if split_idx:
        title += f" | Split {split_idx}"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def cleanup_cuda(model=None, verbose=False):
    """
    Forcefully releases GPU memory used by a model or other CUDA tensors.

    Args:
        model (torch.nn.Module, optional): Model to move to CPU and delete.
        verbose (bool): If True, prints memory usage before and after cleanup.
    """
    if verbose:
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(
            f"[DEBUG] Before cleanup: {allocated:.2f} MB allocated | {reserved:.2f} MB reserved"
        )

    if model is not None:
        model.to("cpu")
        del model

    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(
            f"[DEBUG] After cleanup: {allocated:.2f} MB allocated | {reserved:.2f} MB reserved"
        )
