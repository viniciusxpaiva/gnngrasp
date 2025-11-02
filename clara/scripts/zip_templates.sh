#!/bin/bash

# Caminho base original
BASE_DIR="/home/vinicius/Desktop/deep-grasp/data/templates"

# Caminho para salvar os zips
ZIP_DIR="/home/vinicius/Desktop/deep-grasp/data/templates_zipped"

# Número de threads para paralelizar (ajuste conforme sua máquina)
NUM_THREADS=8

# Encontra todos os arquivos .csv
csv_files=($(find "$BASE_DIR" -name "*.csv" -type f))
total=${#csv_files[@]}

# Função para zipar um arquivo
zip_file() {
    local file="$1"
    local relative_path="${file#$BASE_DIR/}"  # pega caminho relativo
    local zip_target_dir="$ZIP_DIR/$(dirname "$relative_path")"
    local filename=$(basename "$file")

    mkdir -p "$zip_target_dir"  # cria pasta de destino (preservando estrutura)

    local zip_file_path="$zip_target_dir/${filename}.zip"

    if [[ -f "$zip_file_path" ]]; then
        echo "[SKIP] $relative_path já zipado."
        return
    fi

    zip -j "$zip_file_path" "$file" > /dev/null
    echo "[✓] Zipado: $relative_path"
}

export -f zip_file
export BASE_DIR
export ZIP_DIR

echo "[INFO] Serão zipados $total arquivos CSV."
echo ""

# Cria a pasta raiz dos zips se não existir
mkdir -p "$ZIP_DIR"

# Mostra barra de progresso
progress=0
(
    for file in "${csv_files[@]}"; do
        zip_file "$file"
        ((progress++))
        echo "$progress" >&3
    done
) 3> >(awk -v total="$total" '
    {
        percent = int(($1/total)*100)
        bar = ""
        for (i=0; i<percent/2; i++) bar = bar "="
        printf("\r[%s%-50s] %d%% (%d/%d)", bar, "", percent, $1, total) > "/dev/stderr"
    }
')

echo ""
echo "[✓] Finalizado!"
