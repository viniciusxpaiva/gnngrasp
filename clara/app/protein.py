from Bio.PDB import PDBParser, is_aa, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os


class Protein:
    """
    Represents a protein parsed from a PDB file,
    focusing on extracting sequence, node embeddings, and edge embeddings.
    """

    def __init__(self, pdb_id, pdb_path, chain_id=""):
        """
        Initialize the Protein instance.

        Args:
            pdb_id (str): PDB entry identifier.
            pdb_path (str): Path to the local PDB file.
            chain_id (str, optional): Chain identifier. If empty, includes all chains.
        """
        self.pdb_id = pdb_id
        self.pdb_path = pdb_path
        self.chain_id = chain_id
        self.pdb_parser = PDBParser(QUIET=True)

        # Load structure and sequence
        self.structure = self._load_structure()
        if self.structure is None:
            raise ValueError(
                f"[!] Protein structure {pdb_id} with chain {chain_id} could not be loaded."
            )

        self.sequence = self._load_sequence()
        if not self.sequence:
            raise ValueError(f"[!] Empty sequence extracted for {pdb_id}_{chain_id}.")

        # Internal attributes for embeddings
        self._global_embeddings = None
        self._node_embeddings = None
        self._edge_embeddings = None

        # Internal attributes for properties
        self._node_properties = None

        # Determine if the structure and sequence are valid
        self.is_valid = self._check_validity()

    #############################################################
    # Properties
    #############################################################

    @property
    def global_embeddings(self):
        """Returns the global protein embedding (ESM)."""
        return self._global_embeddings

    @global_embeddings.setter
    def global_embeddings(self, value):
        self._global_embeddings = value

    @property
    def node_embeddings(self):
        """Returns the node embeddings for residues."""
        return self._node_embeddings

    @node_embeddings.setter
    def node_embeddings(self, value):
        self._node_embeddings = value

    @property
    def edge_embeddings(self):
        """Returns the edge embeddings for residue pairs."""
        return self._edge_embeddings

    @edge_embeddings.setter
    def edge_embeddings(self, value):
        self._edge_embeddings = value

    @property
    def node_properties(self):
        """Returns the node properties for residues."""
        return self._node_properties

    @node_properties.setter
    def node_properties(self, value):
        self._node_properties = value

    @property
    def edge_properties(self):
        """Returns the edge properties for residues."""
        return self._edge_properties

    @edge_properties.setter
    def edge_properties(self, value):
        self._edge_properties = value

    #############################################################
    # Auxiliary Functions
    #############################################################

    def get_chains(self):
        """Return a list of chains present in the loaded structure."""
        return list(self.structure.get_chains())

    def save_single_model_pdb(self, output_path):
        """
        Saves the current structure (model 0, optionally filtered by chain) to a PDB file.

        Parameters:
            output_path (str): Destination file path.
        """
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(output_path)

    def save_fasta(self, out_dir):
        """
        Save the protein sequence as a .fasta file for BLAST.
        """
        fasta_path = os.path.join(out_dir, f"{self.pdb_id}_{self.chain_id}.fasta")

        record = SeqRecord(
            Seq(self.sequence), id=f"{self.pdb_id}_{self.chain_id}", description=""
        )
        SeqIO.write(record, fasta_path, "fasta")
        self.fasta_files = [fasta_path]

    def _load_structure(self):
        """
        Load the PDB structure and keep only model 0 and the specified chain,
        filtering out HETATM and non-standard residues.
        """
        full_structure = self.pdb_parser.get_structure(self.pdb_id, self.pdb_path)

        # Get model 0 only
        if 0 not in full_structure:
            print(f"[!] Model 0 not found in {self.pdb_id}")
            return None
        model_0 = full_structure[0]

        # If a chain was specified, extract it and filter residues
        if self.chain_id:
            if self.chain_id not in model_0:
                print(f"[!] Chain {self.chain_id} not found in model 0.")
                return None
            filtered_structure = self._extract_single_chain_structure(
                model_0[self.chain_id]
            )
        else:
            # Extract and combine all chains with valid amino acid residues

            builder = StructureBuilder()
            builder.init_structure(self.pdb_id)
            builder.init_model(0)

            for chain in model_0:
                builder.init_chain(chain.id)
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    builder.init_seg("    ")
                    builder.init_residue(
                        residue.resname, residue.id[0], residue.id[1], residue.id[2]
                    )
                    for atom in residue:
                        builder.init_atom(
                            atom.name,
                            atom.coord,
                            atom.bfactor,
                            atom.occupancy,
                            atom.altloc,
                            atom.fullname,
                            atom.serial_number,
                            element=atom.element,
                        )

            filtered_structure = builder.get_structure()

        # Sanity check: ensure we have at least one residue
        if not any(
            is_aa(res, standard=True) for res in filtered_structure.get_residues()
        ):
            print(
                f"[!] No valid amino acid residues found in {self.pdb_id}_{self.chain_id}"
            )
            return None

        return filtered_structure

    def _load_sequence(self):
        """
        Extract the one-letter amino acid sequence from the structure.
        """
        aa_map = {k.upper(): v for k, v in protein_letters_3to1.items()}
        sequence = ""

        for model in self.structure:
            for chain in model:
                if chain.id != self.chain_id and self.chain_id != "":
                    continue
                for residue in chain:
                    if residue.id[0] == " ":  # Only standard residues
                        resname = residue.get_resname().strip().upper()
                        sequence += aa_map.get(resname, "X")
                return sequence  # Return after the correct chain is processed

        return ""

    def _extract_single_chain_structure(self, chain):
        """
        Constructs a new Structure object containing only the specified chain,
        filtering only standard amino acid residues (excluding HETATM).
        """
        builder = StructureBuilder()
        builder.init_structure(self.pdb_id)
        builder.init_model(0)
        builder.init_chain(chain.id)

        for residue in chain:
            # Skip heteroatoms and non-standard residues
            if not is_aa(residue, standard=True):
                continue

            builder.init_seg("    ")  # Default segment
            builder.init_residue(
                residue.resname, residue.id[0], residue.id[1], residue.id[2]
            )
            for atom in residue:
                builder.init_atom(
                    atom.name,
                    atom.coord,
                    atom.bfactor,
                    atom.occupancy,
                    atom.altloc,
                    atom.fullname,
                    atom.serial_number,
                    element=atom.element,
                )

        return builder.get_structure()

    def _check_validity(self):
        """
        Check if the protein is valid:
        - Has a loaded structure.
        - Has a non-empty sequence.
        - Sequence is not composed only of unknown residues (X).
        """
        if self.structure is None:
            return False
        if not self.sequence:
            return False
        if set(self.sequence) == {"X"}:
            return False
        return True
