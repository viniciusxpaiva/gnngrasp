from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import *
from consensus_methods import *
import tempfile
import uuid

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
    data = request.get_json()
    input_string = "input"
    prot_folder = "input"

    if not prot_folder:
        return jsonify({'grasp': [],
                    'puresnet': [], 
                    'deeppocket': [], 
                    'pointsite': [],
                    'p2rank': [],
                    'summary': [],
                    'prot_folder': [],
                    'all_residues': [],
                    'mean_consensus' : [],
                    'max_consensus_percent': [],
                    'ai_prediction' : [],
                    'prot_full_name': []})

    protein_residues = get_all_protein_residues(input_string, prot_folder)

    prot_full_name = "input"

    bsites_grasp = get_prediction_results("../clara/input/predictions.csv")

    out_dir = Path("../clara/input/")
    run_deep_grasp(out_dir)
    #run_deep_grasp()

    return jsonify({'grasp': bsites_grasp,
                    'prot_folder': prot_folder,
                    'all_residues': protein_residues,
                    'prot_full_name': prot_full_name})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


