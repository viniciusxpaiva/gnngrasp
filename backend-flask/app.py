from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import *
from consensus_methods import *
import tempfile
import uuid

from pathlib import Path
import shutil
from datetime import datetime
import os


app = Flask(__name__)
CORS(app)

# Route for serving React app

# Route for API data
@app.route('/data1')
def get_time():
    return {
        'Name': "geek",
        "Age": "25",
        "Date": "some date",
        "programming": "python"
    }

@app.route('/prot_folder', methods=['POST'])
def prot_folder():
    data = request.get_json()
    search_string = data.get('searchString', '')

    prot_folder = search_PDB(search_string)

    return jsonify({'prot_folder': prot_folder})

@app.route('/process', methods=['POST'])
def process_string():
    uploaded_file = request.files.get('pdbFile')
    form_data_code = request.form.get('pdbCode', '').strip().lower()
    form_job_id = request.form.get('jobId', '').strip()
    data = request.get_json(silent=True) or {}
    job_id_requested = (form_job_id or data.get('jobId') or '').strip()

    pdb_code = (form_data_code or data.get('pdbCode') or '').strip().lower()
    if uploaded_file and uploaded_file.filename and not pdb_code:
        pdb_code = Path(uploaded_file.filename).stem.strip().lower() or pdb_code

    # Helper para checar status de job existente
    if job_id_requested:
        job_id = job_id_requested
        job_dir = Path("../clara/results") / f"output_{job_id}"
        job_dir_front = Path("../frontend-react/public/pdbs/") / f"output_{job_id}"

        if not job_dir.exists():
            return jsonify({'error': 'JOB_NOT_FOUND'}), 404

        existing_pdb = next(job_dir.glob("*.pdb"), None)
        if not existing_pdb:
            return jsonify({'error': 'PDB_NOT_FOUND'}), 404

        pdb_code = existing_pdb.stem.lower()
        pdb_stem_original = existing_pdb.stem  # respeita maiúsculas/underscores do arquivo
        job_dir_front.mkdir(parents=True, exist_ok=True)
        if not (job_dir_front / existing_pdb.name).exists():
            shutil.copy2(existing_pdb, job_dir_front / existing_pdb.name)

        existing_csv = next(job_dir.glob("*_prediction.csv"), None)
        if existing_csv:
            if not (job_dir_front / existing_csv.name).exists():
                shutil.copy2(existing_csv, job_dir_front / existing_csv.name)

            protein_residues = get_all_protein_residues(pdb_stem_original, job_dir)
            prot_full_name = pdb_stem_original
            bsites_grasp = get_prediction_results(existing_csv, existing_pdb)

            print(bsites_grasp)

            return jsonify({
                'status': 'DONE',
                'grasp': bsites_grasp,
                'prot_folder': str(job_dir),
                'all_residues': protein_residues,
                'prot_full_name': prot_full_name,
                'job_id': job_id,
                'pdb_code': pdb_stem_original,
            })
        else:
            pid_file = job_dir / "job.pid"
            pid_running = False
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    if pid > 0 and Path(f"/proc/{pid}").exists():
                        pid_running = True
                except Exception:
                    pid_running = False

            if pid_running:
                return jsonify({
                    'status': 'RUNNING',
                    'job_id': job_id,
                    'pdb_code': pdb_code,
                }), 202
            else:
                return jsonify({
                    'error': 'JOB_FAILED',
                    'status': 'FAILED',
                    'job_id': job_id,
                    'pdb_code': pdb_stem_original,
                }), 500

            return jsonify({
                'status': 'RUNNING',
                'job_id': job_id,
                'pdb_code': pdb_code,
            }), 202

    if not pdb_code and not (uploaded_file and uploaded_file.filename):
        return jsonify({'error': 'MISSING_PDB_CODE'}), 400

    # 1) Cria um ID único para esse input
    job_id = datetime.now().strftime("%y%m%d%H%M") + "-" + uuid.uuid4().hex[:4]

    # 2) Cria diretório único (input e output no mesmo lugar)
    job_dir = Path("../clara/results") / f"output_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    # 2) Cria diretório único (input e output no mesmo lugar)
    job_dir_front = Path("../frontend-react/public/pdbs/") / f"output_{job_id}"
    job_dir_front.mkdir(parents=True, exist_ok=True)

    # 3) Usa o PDB enviado ou faz download se for código
    if uploaded_file and uploaded_file.filename:
        normalized_code = (pdb_code or Path(uploaded_file.filename).stem).strip().lower()
        pdb_filename = f"{normalized_code}.pdb"
        pdb_path = job_dir / pdb_filename
        uploaded_file.save(pdb_path)
        shutil.copy2(pdb_path, job_dir_front / pdb_path.name)
        pdb_code = Path(pdb_filename).stem.lower()
    else:
        try:
            pdb_path = download_pdb_from_rcsb(pdb_code, job_dir)
            shutil.copy2(pdb_path, job_dir_front / pdb_path.name)
        except PdbNotFoundError:
            return jsonify({'error': 'PDB_NOT_FOUND'}), 404
        except Exception as e:
            print("Error downloading PDB:", e)
            return jsonify({'error': 'INTERNAL_ERROR'}), 500

    # 4) Pasta de output pro Deep-GRaSP (também usando o ID único)
    out_dir = job_dir

    # Dispara processamento em background e retorna status de execução
    pid = start_deep_grasp_async(pdb_code, out_dir)
    try:
        (out_dir / "job.pid").write_text(str(pid))
    except Exception:
        pass

    print(f"Job {job_id} iniciado (PID {pid}) para {pdb_code}")

    return jsonify({
        'status': 'RUNNING',
        'job_id': job_id,
        'pdb_code': pdb_code,
        'pid': pid,
    }), 202

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
