import os
import json

dict_diseases = {'Mycobacterium ulcerans':'Buruli ulcer', 'Trypanosoma cruzi':'Chagas disease', 'Dracunculus medinensis':'Dracunculiasis', 'Leishmania infantum':'Leishmaniasis',
				 'Wuchereria bancrofti':'Lymphatic filariasis', 'Mycobacterium leprae':'Leprosy', 'Plasmodium falciparum':'Malaria', 'Onchocerca volvulus':'Onchocerciasis',
				 'Schistosoma mansoni':'Schistosomiasis', 'Trichuris trichiura':'Trichuriasis','tmpFolderAux':'Mycobacterium ulcerans'}

folder_path = '../frontend-react/public/pdbs/'

proteomes = os.listdir(folder_path)

file_path = 'output.json'

total_list = []

for proteome in proteomes:
	prot_list = os.listdir(folder_path + proteome)
	for prot in prot_list:
		uniprot = prot.split('-')[1]
		species = proteome.replace('_', ' ')
		disease = dict_diseases[species]
		link = '/results/' + uniprot


		row_dict = {'uniprot':uniprot, 'species':species, 'disease': disease, 'results':link}
		total_list.append(row_dict)


# Write the list to the file
with open(file_path, 'w') as file:
    json.dump(total_list, file, indent=2)

print(f'The data has been written to {file_path}')
