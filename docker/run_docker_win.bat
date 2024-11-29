@echo off

:: Obtener el directorio donde se encuentra este archivo .bat
set SCRIPT_DIR=%~dp0

:: Quitar la barra invertida final del directorio
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

:: Definir las rutas de entrada y salida basadas en el directorio del script
set INPUT_DIR=%SCRIPT_DIR%\data\input
set OUTPUT_DIR=%SCRIPT_DIR%\data\output

:: Ejecutar el contenedor Docker
docker run --rm -v "%INPUT_DIR%:/data/input" -v "%OUTPUT_DIR%:/data/output" tp2_alsop_hachen