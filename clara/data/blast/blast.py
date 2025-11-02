import os


class BLAST:
    def __init__(self, blast_bin_dir, database_name="biolip"):
        """
        Args:
            blast_bin_dir (str): Path to directory containing BLAST+ binaries (e.g. blastp).
            database_name (str): Name of the BLAST-formatted database (no file extension).
        """
        self.blast_bin_dir = blast_bin_dir
        self.database = database_name

    def run(self, query, max_templates=None, output_dir=None):
        """
        Run BLAST for the given Protein object, and select templates.

        Args:
            query (Protein): Protein object with .fasta_files populated.
            max_templates (int, optional): If provided, selects the top-N templates by score.
            output_dir (str, optional): Directory to save .out files.

        Returns:
            set: Set of selected template identifiers.
        """
        if not query.fasta_files:
            raise ValueError("[ERROR] No FASTA files provided in Protein object.")

        e_value = 0.001
        max_e_value = 100
        all_hits = []  # List of (template_id, score)
        selected_templates = set()

        while e_value <= max_e_value:
            for fasta_path in query.fasta_files:
                fasta_filename = os.path.basename(fasta_path)
                out_path = (
                    os.path.join(output_dir, fasta_filename)
                    if output_dir
                    else fasta_path
                ).replace(".fasta", ".out")

                blast_command = (
                    f"{os.path.join(self.blast_bin_dir, 'blastp')}"
                    f" -db {os.path.join(self.blast_bin_dir, 'biolip.fasta')}"
                    f" -evalue {e_value}"
                    f" -query {fasta_path}"
                    f" -outfmt '6 delim=,'"
                    f" -out {out_path}"
                )

                exit_code = os.system(blast_command)
                if exit_code != 0:
                    print(f"[ERROR] BLAST failed for {fasta_path}")
                    continue

                result = BlastResult(out_path)

                if max_templates is not None:
                    all_hits.extend(result.get_matches())
                else:
                    selected_templates |= result.get_result()

            if max_templates is not None and len(all_hits) >= max_templates:
                break

            if not max_templates and selected_templates:
                break

            e_value *= 10

        # If top-n mode
        if max_templates is not None:
            if not all_hits:
                print("[WARN] No templates found with BLAST.")
                return set()

            hit_dict = {}
            for tid, score in all_hits:
                if tid not in hit_dict or score > hit_dict[tid]:
                    hit_dict[tid] = score

            selected_templates = sorted(
                hit_dict.items(), key=lambda x: x[1], reverse=True
            )[:max_templates]
            return set(t[0] for t in selected_templates)

        # Default e-value case
        return selected_templates


class BlastResult:
    """
    Parses a BLAST output file and extracts matched subject identifiers and scores.
    """

    def __init__(self, file_path):
        self.matches = []
        try:
            with open(file_path) as file:
                for line in file:
                    fields = line.strip().split(",")
                    if len(fields) >= 12:
                        template_id = fields[1]
                        bit_score = float(fields[11])
                        self.matches.append((template_id, bit_score))
        except FileNotFoundError:
            print(f"[ERROR] BLAST result file not found: {file_path}")

    def get_matches(self):
        """Returns list of (template_id, score)."""
        return self.matches

    def get_result(self):
        """Returns set of template IDs only (for backward compatibility)."""
        return set(template_id for template_id, _ in self.matches)

    def __len__(self):
        return len(self.matches)


if __name__ == "__main__":
    print("[!] BLAST module ready.")
