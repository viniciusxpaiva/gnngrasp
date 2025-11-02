import os

# Caminhos para suas pastas
edge_dir = "../data/prop_templates/edge_properties"
node_dir = "../data/embd_templates/node_embeddings"


# Função para extrair os prefixos (6 primeiros caracteres)
def get_prefixes(path, suffix_to_remove):
    return {
        filename[:6]
        for filename in os.listdir(path)
        if filename.endswith(".zip") and suffix_to_remove in filename
    }


# Extrai os prefixos dos arquivos
edge_prefixes = get_prefixes(edge_dir, "edge_properties")
node_prefixes = get_prefixes(node_dir, "node_embeddings")

# Arquivos que existem em edge, mas não em node
only_in_edge = sorted(edge_prefixes - node_prefixes)

# Arquivos que existem em node, mas não em edge
only_in_node = sorted(node_prefixes - edge_prefixes)

# Exibir resultados
print(f"Arquivos só na pasta '{edge_dir}': {len(only_in_edge)}")
for prefix in only_in_edge:
    print(prefix)

print(f"\nArquivos só na pasta '{node_dir}': {len(only_in_node)}")
for prefix in only_in_node:
    print(prefix)
