#!/bin/bash

# Nome do arquivo original
input="full_biolip_binding_data.txt"

# Número de divisões
n=10

# Total de linhas
total_lines=$(wc -l < "$input")

# Linhas por arquivo (arredondando para cima)
lines_per_file=$(( (total_lines + n - 1) / n ))

# Dividir o arquivo
split -d -l "$lines_per_file" "$input" temp_split_

# Renomear os arquivos
i=1
for file in temp_split_*; do
    mv "$file" "${i}_full_biolip_binding_data.txt"
    i=$((i + 1))
done
