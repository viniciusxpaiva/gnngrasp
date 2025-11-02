import pymol
from search_utils import *

def create_pymol_session(prot_name):

	# Initialize PyMOL in command-line mode
	pymol.finish_launching(['pymol', '-c'])

	# Load the PDB file
	pymol.cmd.load("AF-" + prot_name + "-F1-model_v4.pdb")

	# Apply the "spectrum b" command to color the protein by B-factor
	pymol.cmd.spectrum("b", "blue_white_red", "AF-A4HXH5-F1-model_v4")

	# Save the session
	pymol.cmd.save(prot_name + "_pymol_session.pse")


def color_bsites_pymol(prot_name, bsites, predictor):
			
	def color_residues2(bsites):
	    pymol.cmd.color("grey", "all")
	    colors = ["palladium", "olive", "firebrick", "nickel", "antimony", "zirconium", "boron", "cerium", "neodymium", "manganese"]
	    for idx, bsite in enumerate(bsites):
	        selection_name = f"binding_site_{idx}"
	        binding_site_selection = []
	        for chain, res_name, res_num in bsite:
	            pymol_color = pymol.cmd.get_color_index(f"{colors[idx % len(colors)]}")
	            pymol.cmd.color(pymol_color, f"chain {chain} and resi {res_num}")
	            pymol.cmd.show("sticks", f"chain {chain} and resi {res_num}")
	            pymol.cmd.hide("cartoon", f"chain {chain} and resi {res_num}")
	            binding_site_selection.append(f"chain {chain} and resi {res_num}")
	        pymol.cmd.select(selection_name, " or ".join(binding_site_selection))


	# Load your protein structure
	pymol.cmd.load("AF-" + prot_name + "-F1-model_v4.pdb")

	# Color the specified residues pink
	color_residues2(bsites)

	# Color the rest of the protein white
	#pymol.cmd.color("white", "all and not resi " + "+".join([res[2] for res in residue_list[0]]))

	# Save the modifications as a PyMOL session
	pymol.cmd.save(prot_name + "_" + predictor + "_sites_pymol_session.pse")


def find_occurrence(res_to_find, residues_list):

	for residue in residues_list:
		res_info = residue[0]
		# Se for usar a quantidade de sítios e não de preditores, usar quant_preds = residue[2]
		quant_preds = len(residue[1])
		if res_to_find == res_info:
			return quant_preds

	return 0.00


def set_normalized_bfactor(max_temp_factor, modified_lines, prot_name):

	new_temp_factor_lines = []
	for line in modified_lines:
		if line[:4] == 'ATOM':
			new_temp_factor_norm = round(((float(line[60:66].replace(' ',''))/max_temp_factor)*99.99),2)
			print(new_temp_factor_norm, max_temp_factor)
			line = line[:61] + f"{new_temp_factor_norm:.2f}" + line[66:]

		new_temp_factor_lines.append(line)

	with open("Norm_AF-" + prot_name + "-F1-model_v4.pdb", "w") as modified_file:
		modified_file.writelines(new_temp_factor_lines)

def change_protein_bfactor(prot_name, residues_list, num_pred_found):
	max_temp_factor = 0
	#print(residues_list)

	modified_lines = []


	protein_file = open("AF-" + prot_name + "-F1-model_v4.pdb", "r")
	pdb_lines = protein_file.readlines()

	for line in pdb_lines:
		if line[:4] == 'ATOM':
			#print(line.replace('\n',''))
			
			res_name = line[17:20]
			chain = line[21]
			seq_num = line[22:26].replace(' ','')
			temp_factor = line[60:66].replace(' ','')
			occur = find_occurrence([chain, res_name, seq_num], residues_list)

			new_temp_factor = round((100/num_pred_found)*occur,2)
			if new_temp_factor > max_temp_factor:
				max_temp_factor = new_temp_factor			

			line = line[:61] + f"{new_temp_factor:.2f}" + line[66:]
			
		modified_lines.append(line)

	set_normalized_bfactor(max_temp_factor, modified_lines, prot_name)

	with open("New_AF-" + prot_name + "-F1-model_v4.pdb", "w") as modified_file:
		modified_file.writelines(modified_lines)



if __name__ == '__main__':

	input_string = "A4HXH5"

	bsites_grasp = grasp_search(input_string)
	bsites_puresnet = puresnet_search(input_string)
	bsites_gass = []
	bsites_deeppocket = deeppocket_search(input_string)
	bsites_pointsite = pointsite_search(input_string)
	bsites_p2rank = p2rank_search(input_string)
	
	color_bsites_pymol("A4HXH5", bsites_grasp, "GRaSP")
	color_bsites_pymol("A4HXH5", bsites_puresnet, "PUResNet")
	color_bsites_pymol("A4HXH5", bsites_deeppocket, "DeepPocket")
	color_bsites_pymol("A4HXH5", bsites_pointsite, "PointSite")
	color_bsites_pymol("A4HXH5", bsites_p2rank, "p2Rank")