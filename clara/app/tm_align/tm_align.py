import subprocess
import os
import re
from typing import List, Tuple


class TMAlign:
    def __init__(self, tmalign_path: str = "TMalign"):
        self.tmalign_path = tmalign_path

    def run_tmalign(self, input_pdb: str, template_pdb: str) -> float:
        """
        Executa o TM-align entre a proteína de entrada e o template.
        Retorna o TM-score normalizado pela proteína de entrada (Chain_1).
        """
        try:
            result = subprocess.run(
                [self.tmalign_path, input_pdb, template_pdb],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
            )

            # Debug da saída
            output = result.stdout

            # Extrair a linha correta do TM-score
            match = re.search(
                r"TM-score=\s*([\d.]+)\s+\(if normalized by length of Chain_1", output
            )
            if match:
                return float(match.group(1))

            # Fallback: pega o primeiro TM-score da saída, se for o caso
            fallback = re.findall(r"TM-score=\s*([\d.]+)", output)
            if fallback:
                return float(fallback[0])

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run TM-align for {template_pdb}: {e.stderr}")

        return 0.0

    def align_and_rank_templates(
        self,
        input_pdb: str,
        templates_dir: str,
        top_n: int = 5,
        file_extension: str = ".pdb",
    ) -> List[Tuple[str, float]]:
        """
        Alinha o input_pdb com todos os templates no diretório e retorna os top N templates mais similares.
        """
        results = []
        cont = 0
        for template_file in os.listdir(templates_dir):
            if cont % 100 == 0:
                print(f"Processed {cont}/{len(os.listdir(templates_dir))}")
            cont += 1
            if template_file.endswith(file_extension):
                template_path = os.path.join(templates_dir, template_file)
                score = self.run_tmalign(input_pdb, template_path)
                results.append((template_file, score))

        # Ordenar por TM-score decrescente
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
