import os
import glob
import pandas as pd

BENDERDB_DATA_PATH = '/home/vinicius/binding_results/'
BACKEND_PATH = '/var/www/benderdb/backend-flask/'

def format_bsite_string(bsite_string):
	items = bsite_string.split(',')
	processed_result = [item.split('_') for item in items]
	return processed_result

def create_dataframe(prot_name, data, predictor):
	rows = []
	site_num = 0
	for inner_list in data:
		residues = ','.join([f'{x[1]}_{x[2]}_{x[0]}' for x in inner_list])
		rows.append([predictor, site_num, residues])
		site_num += 1
	if rows:
		df = pd.DataFrame(rows, columns=['Predictor', 'Site', 'Residues'])
		df.to_csv(BENDERDB_DATA_PATH + "results/" +  prot_name + '_' + predictor + '_results.csv', index=False)


def create_download_files(prot_name, bsites_grasp, bsites_puresnet, bsites_gass, bsites_deeppocket, bsites_pointsite, bsites_p2rank):
	create_dataframe(prot_name, bsites_grasp, "GRaSP")
	create_dataframe(prot_name, bsites_puresnet, "PUResNet")
	create_dataframe(prot_name, bsites_deeppocket, "DeepPocket")
	create_dataframe(prot_name, bsites_pointsite, "PointSite")
	create_dataframe(prot_name, bsites_p2rank, "P2Rank")


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



if __name__ == '__main__':
	pdbs_folder_path = '/home/vinicius/content-bender/pdbs/'
	
	
	for proteome_folder in os.listdir(pdbs_folder_path):
		print("Proteome:", proteome_folder)
		cont = 0
		all_pdbs_in_proteome_folder = os.listdir(pdbs_folder_path + proteome_folder)
		all_pdbs_in_proteome_folder = [ele.split('-')[1] for ele in all_pdbs_in_proteome_folder]

		for input_string in all_pdbs_in_proteome_folder:
			cont += 1
			if cont % 500 == 0:
				print('Creating results:', cont)
			bsites_grasp = grasp_search(input_string)
			bsites_puresnet = puresnet_search(input_string)
			bsites_gass = []
			bsites_deeppocket = deeppocket_search(input_string)
			bsites_pointsite = pointsite_search(input_string)
			bsites_p2rank = p2rank_search(input_string)
			create_download_files(input_string, bsites_grasp, bsites_puresnet, bsites_gass, bsites_deeppocket, bsites_pointsite, bsites_p2rank)
	print("Results finish!")
