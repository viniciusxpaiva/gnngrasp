import pandas as pd
import os
import glob
from pathlib import Path
import subprocess

BACKEND_PATH = ''
#BACKEND_PATH = '/var/www/benderdb/backend-flask/'

BENDERDB_DATA_PATH = '../frontend-react/public/'
#BENDERDB_DATA_PATH = '/var/www/benderdb-data/'

CONDA_ENV_NAME = "deep-grasp"                     # ex.: "deepgrasp"
DEEP_GRASP_WORKDIR = "/home/vinicius/Desktop/deep-grasp/"  # pasta onde vive o deep-grasp.py
DEEP_GRASP_SCRIPT  = "deep-grasp.py"           # nome do script
CSV_RELATIVE_PATH  = "saida-flask.csv"   # caminho padrão de saída relativo ao workdir ou ao -o



def get_prediction_results(input_file):
	df = pd.read_csv(input_file)

	# filtra apenas as linhas com predicted_label = 1
	df_filtered = df[df["predicted_label"] == 1]

	# transforma o residue_id no formato desejado
	# exemplo: "TRP_32_A" → ['A', 'TRP', '32']
	def split_residue(residue_id):
	    resname, resnum, chain = residue_id.split("_")
	    return [chain, resname, resnum]

	# aplica a função
	residue_lists = [split_residue(rid) for rid in df_filtered["residue_id"]]

	# coloca dentro de uma lista externa
	result = [residue_lists]

	return result


def run_deep_grasp(output_dir: Path, extra_args=None, timeout_sec=1800):
    """
    Executa o pipeline via conda e retorna o caminho do CSV gerado.
    - output_dir: diretório de saída (será passado via -o)
    - extra_args: lista com argumentos extras a repassar ao script (opcional)
    """
    extra_args = extra_args or []

    # Garante que a pasta exista
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comando: conda run -n <env> python deep-grasp.py -o <saida> <extra_args...>
    cmd = [
        "conda", "run", "-n", CONDA_ENV_NAME,
        "python", DEEP_GRASP_SCRIPT,
        "-o", str(output_dir),
        *extra_args
    ]

    # Executa no diretório do projeto
    proc = subprocess.run(
        cmd,
        cwd=DEEP_GRASP_WORKDIR,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,   # vamos checar manualmente
        env={**os.environ},  # herda env atual
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"deep-grasp falhou (code={proc.returncode}). "
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # CSV esperado: geralmente algo como <output_dir>/predictions.csv.
    # Se o script sempre escreve "predictions.csv" dentro do diretório passado em -o:
    csv_path = Path(output_dir) / "predictions.csv"
    if not csv_path.exists():
        # fallback: se seu script escreve em outro lugar/caminho, ajuste aqui
        alt = Path(DEEP_GRASP_WORKDIR) / CSV_RELATIVE_PATH
        if alt.exists():
            csv_path = alt
        else:
            raise FileNotFoundError(
                f"CSV de saída não encontrado em {csv_path} nem em {alt}. "
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

    return csv_path, proc.stdout  # devolve também logs se quiser inspecionar


def get_protein_full_name(prot_name, pdb_folder):
	pdb_name = '/AF-' + prot_name.upper() + '-F1-model_v4.pdb'
	file = open(BENDERDB_DATA_PATH + 'pdbs/' + pdb_folder + pdb_name)
	file_lines = file.readlines()
	file.close()

	reading_name = False
	prot_full_name = ""
	for line in file_lines:
		if line[:6] == 'COMPND':
			if line[11:19] == 'MOLECULE':
				reading_name = True
				prot_full_name += line[21:].replace('\n','')
			elif line[11:16] == 'CHAIN':
				break
			elif reading_name:
				prot_full_name += line[11:].replace('\n','')
	
	return prot_full_name.split(';')[0]


def search_PDB(search_string):
	pdb_folder = BENDERDB_DATA_PATH + 'pdbs/'

	pdb_name = 'AF-' + search_string.upper() + '-F1-model_v4.pdb'
	
	proteome_folders = os.listdir(pdb_folder)

	for proteome in proteome_folders:
		if pdb_name in os.listdir(pdb_folder + proteome):
			return proteome

	return ''
	

def format_bsite_string(bsite_string):
	items = bsite_string.split(',')
	processed_result = [item.split('_') for item in items]
	return processed_result


def get_all_protein_residues(prot_name, prot_folder):
	pdb_folder = BENDERDB_DATA_PATH + 'pdbs/' + prot_folder + '/'
	#pdb_name = 'AF-' + prot_name.upper() + '-F1-model_v4.pdb'
	protein_file = open(pdb_folder + prot_name + ".pdb", "r")
	pdb_lines = protein_file.readlines()

	residues_list = []

	for line in pdb_lines:
		if line[:4] == 'ATOM':
			new_res = []
			new_res.append(line[21])
			new_res.append(line[17:20])
			new_res.append(line[22:26].replace(' ',''))
			residues_list.append(new_res)

	unique_lists = list(map(list, set(map(tuple, residues_list))))
	sorted_list = sorted(unique_lists, key=lambda x: int(x[2]))
	return sorted_list

def count_common_residues(total_res, unique_total_res):
	'''
	Function to count occurrence of unique residues in a list.
	Params: 
		- list of total residues found in sites/pockets (including duplicates)
		- list of unique residues found
	Return:
		- list with sorted residues by number and its occurrence in sites/pockets
	'''
	
	# List to count number of occurrence for each residue
	len_list = [0] * len(unique_total_res)
	
	# Loop through unique list and total list to count number of occurrences
	for i in range(0, len(unique_total_res)):
		for j in range(0, len(total_res)):
			if unique_total_res[i] == total_res[j]:
				len_list[i] += 1

	# Append number of occurrences to unique list
	unique_with_count = unique_total_res
	for i in range(0, len(len_list)):
		unique_with_count[i].append(len_list[i])

	# Sort result list by occurrence
	sorted_unique_with_count = sorted(unique_with_count, key=lambda x: x[-1], reverse=True)

	return sorted_unique_with_count


def get_intersections(res_pred_list):
	'''
	Function to get unique residues and all predictors that found it
	Params:
		- list with residues and its predictor (residues can be duplicate)
	Return:
		- list with unique residues and all predictors that found it
		- format: [[unique residue info], [predictors that found that residue]]
	'''
	result_dict = {}

	
	# Convert input list to dict
	for inner_list in res_pred_list:
	    key = tuple(inner_list[0])  # Convert the list to a tuple to make it hashable
	    value = inner_list[1]
	    
	    if key not in result_dict:
	        result_dict[key] = []

	    result_dict[key].append(value)

	# Iterate through dict and concatenate values into unique keys
	result_list = [[list(key), '-'.join(values)] for key, values in result_dict.items()]

	# Create list of lists that contains unique residues and its predictors
	intersection_list = []
	for r in result_list:
		tmp = []
		tmp.extend([r[0], list(set(r[1].split('-')))])
		intersection_list.append(tmp)

	return intersection_list



def process_intersection_data(bsite_pred_list, unique_res_list):
    '''
    Function to process intersection of residues between predictors
    Params:
    	- list of sites/pockets of each predictors (with duplicates)
    	- list of unique residues (with occurrence - will not be needed)
    Return:
    	- intersection list format: [[unique residue info], [predictors that found that residue]]
    '''
    predictors_order = ['GRaSP', 'PUResNet', 'GASS', 'DeepPocket', 'PointSite', 'P2Rank']

    res_pred_list = []

    # Delete number of occurrence of each residue (not needed)
    #modified_list = [[item[0], item[1], item[2]] for item in unique_res_list]
    
    # Loop through predictors and their sites to get a list with residue info and its predictor
    for i in range (len(predictors_order)):
    	for all_sites in bsite_pred_list[i]:
    		for site in all_sites:
    			for uni in unique_res_list:
	    			if site[:3] == uni[:3]:
	    				tmp = []
	    				tmp.extend([site[:3], predictors_order[i], uni[3]])
	    				res_pred_list.append(tmp)

    


    sorted_list_by_seq = sorted(res_pred_list, key=lambda x: int(x[0][2]))

    # Call function to get unique residues and all predictors that found it
    intersection_list = get_intersections(sorted_list_by_seq)


    for i in intersection_list:
    	for s in sorted_list_by_seq:
    		if i[0] == s[0]:
    			i.append(s[2])
    			break

    return intersection_list


def build_summary(bsites_grasp, bsites_puresnet, bsites_gass, bsites_deeppocket, bsites_pointsite, bsites_p2rank):
	'''
	Function that retrieves summary data which is:
		1 - Number of total sites found in each binding site/pocket of each predictor
		2 - List of most common residues (all residues ordered by occurrence)
		3 - Number of predictors that have at least 1 binding site/pocket found
	Return:
		- list format: [num_total_sites, num_unique_res, [list of most common residues], num_pred_found]
	'''

	total_res = [] # List of all residues, not grouped by binding site or predictor
	unique_total_res = [] # List of unique residues from all binding sites, not grouped by binding site or predictor
	total_sites = [] # List of all binding sites, not grouped by predictor
	total_sites.extend([bsites_grasp, bsites_puresnet, bsites_gass, bsites_deeppocket, bsites_pointsite, bsites_p2rank])

	num_total_sites = 0
	num_pred_found = 0

	# Item number 1 of function description
	for site in total_sites:
		if site != []:
			num_pred_found += 1
	
	# Create list of all residues found
	for pred_sites in total_sites:
		num_total_sites += len(pred_sites)
		for site in pred_sites:
			for res in site:
				total_res.append(res)
	
	# Get only unique residues
	for elem in total_res:
	    if elem not in unique_total_res:
	        unique_total_res.append(elem)

	num_unique_res = len(unique_total_res)

	# Call function to count occurrence of unique residues
	sorted_unique_with_count = count_common_residues(total_res, unique_total_res)
	
	# Sort list of unique residues by sequence number
	sorted_list_by_seq = sorted(sorted_unique_with_count, key=lambda x: int(x[2]))

	intersection_list = process_intersection_data([bsites_grasp, bsites_puresnet, bsites_gass, bsites_deeppocket, bsites_pointsite, bsites_p2rank], sorted_list_by_seq)

	intersection_list = sorted(intersection_list, key=lambda x: int(x[2]), reverse=True)

	return [num_total_sites, num_unique_res, intersection_list, num_pred_found]


def grasp_search(prot_name):
	'''
	Function to handle search for GRaSP results
	'''
	prot_name = prot_name.upper()
	file_path = BACKEND_PATH + 'data/grasp/'
	file_name = 'GRaSP_Concatenated_Sites.csv'

	df = pd.read_csv(file_path + file_name)
	pd.set_option('display.max_colwidth', None)

	matches = df.loc[df['Protein'].str.contains(prot_name), 'Binding_Site']

	if matches.empty:
		return []

	matches_list = matches.to_string(index=False).replace(' ','').split('\n')

	result_list = []
	
	for blist in matches_list:
		result_list.append(format_bsite_string(blist))
	
	return result_list


def puresnet_search(prot_name):
	'''
	Function to handle search for PUResNet results
	'''
	prot_name = prot_name.upper()
	file_path = BACKEND_PATH + 'data/puresnet/'
	file_name = 'PUResNet_Concatenated_Sites.csv'

	df = pd.read_csv(file_path + file_name)
	pd.set_option('display.max_colwidth', None)

	matches = df.loc[df['Protein'].str.contains(prot_name), 'Binding_Site']

	if matches.empty:
		return []

	pd.set_option('display.max_colwidth', None)

	matches_list = matches.to_string(index=False).replace(' ','').split('\n')

	result_list = []
	
	for blist in matches_list:
		result_list.append(format_bsite_string(blist))
	
	return result_list


def p2rank_search(prot_name):
	'''
	Function to handle search for p2Rank results
	'''
	prot_name = prot_name.upper()
	file_path = BACKEND_PATH + 'data/p2rank/'
	file_name = 'p2Rank_Concatenated_Sites.csv'

	df = pd.read_csv(file_path + file_name)
	pd.set_option('display.max_colwidth', None)

	matches = df.loc[df['Protein'].str.contains(prot_name), 'Binding_Site']

	if matches.empty:
		return []

	matches_list = matches.to_string(index=False).replace(' ','').split('\n')

	result_list = []

	for blist in matches_list:
		result_list.append(format_bsite_string(blist))

	return result_list

def pointsite_search(prot_name):
	'''
	Function to handle search for PointSite results
	'''
	prot_name = prot_name.upper()
	file_path = BACKEND_PATH + 'data/pointsite/'
	file_name = 'PointSite_Concatenated_Sites.csv'

	df = pd.read_csv(file_path + file_name)
	pd.set_option('display.max_colwidth', None)

	matches = df.loc[df['Protein'].str.contains(prot_name), 'Binding_Site']

	if matches.empty:
		return []

	matches_list = matches.to_string(index=False).replace(' ','').split('\n')

	result_list = []
	
	for blist in matches_list:
		result_list.append(format_bsite_string(blist))

	return result_list

def deeppocket_search(prot_name):
	'''
	Function to handle search for DeepPocket results
	'''
	prot_name = prot_name.upper()
	file_path = BACKEND_PATH + 'data/deeppocket/'
	file_name = 'DeepPocket_Concatenated_Sites.csv'

	df = pd.read_csv(file_path + file_name)
	pd.set_option('display.max_colwidth', None)

	matches = df.loc[df['Protein'].str.contains(prot_name), 'Binding_Site']

	if matches.empty:
		return []

	matches_list = matches.to_string(index=False).replace(' ','').split('\n')

	result_list = []
	
	for blist in matches_list:
		result_list.append(format_bsite_string(blist))
	
	return result_list
