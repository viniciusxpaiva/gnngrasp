import pandas as pd

		
def format_bsite_string(bsite_string):
	items = bsite_string.split(',')
	processed_result = [item.split('_') for item in items]
	return processed_result

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
	file_path = '../backend-flask/data/grasp/'
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
	file_path = '../backend-flask/data/puresnet/'
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
	file_path = '../backend-flask/data/p2rank/'
	file_name = 'p2Rank_Concatenated_Sites.csv'

	df = pd.read_csv(file_path + file_name)
	pd.set_option('display.max_colwidth', None)

	matches = df.loc[df['Protein'].str.contains(prot_name), 'Binding_Site']

	if matches.empty:
		return []

	matches_list = matches.to_string(index=False).replace(' ','').split('\n')

	result_list = []

	for blist in matches_list:
		#print(blist)
		result_list.append(format_bsite_string(blist))

	return result_list

def pointsite_search(prot_name):
	'''
	Function to handle search for PointSite results
	'''
	prot_name = prot_name.upper()
	file_path = '../backend-flask/data/pointsite/'
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
	file_path = '../backend-flask/data/deeppocket/'
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

	input_string = "A4HXH5"

	bsites_grasp = grasp_search(input_string)
	bsites_puresnet = puresnet_search(input_string)
	bsites_gass = []
	bsites_deeppocket = deeppocket_search(input_string)
	bsites_pointsite = pointsite_search(input_string)
	bsites_p2rank = p2rank_search(input_string)
	#create_pymol_session("A4HXH5")
	

	#print(bsites_p2rank)

	#summary_content = build_summary(bsites_grasp, bsites_puresnet, bsites_gass, bsites_deeppocket, bsites_pointsite, bsites_p2rank)


	#change_protein_bfactor(input_string, summary_content[2], summary_content[3])