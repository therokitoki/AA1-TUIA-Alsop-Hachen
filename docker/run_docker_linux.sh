#!/bin/bash

# Obtener el directorio donde se encuentra este archivo .sh
SCRIPT_DIR=$(dirname "$0")

# Definir las rutas de entrada y salida basadas en el directorio del script
INPUT_DIR="$SCRIPT_DIR/data/input"
OUTPUT_DIR="$SCRIPT_DIR/data/output"

# Ejecutar el contenedor Docker
docker run --rm -v "$INPUT_DIR:/data/input" -v "$OUTPUT_DIR:/data/output" tp2_alsop_hachen