# Deployment del Modelo en Docker

Este repositorio contiene todo lo necesario para realizar el deployment de un modelo de Machine Learning usando Docker. El objetivo es ejecutar un modelo preentrenado y realizar inferencia sobre datos en formato CSV.

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalados los siguientes programas en tu máquina:

### 1. Docker
Docker es necesario para construir y ejecutar el contenedor donde se ejecutará el modelo.  
- [Instrucciones de instalación de Docker para Windows](https://docs.docker.com/docker-for-windows/install/)
- [Instrucciones de instalación de Docker para Linux](https://docs.docker.com/engine/install/)

### 2. Python (opcional, para ejecutar inferencia sin Docker)
Si prefieres ejecutar la inferencia fuera de Docker, también necesitarás instalar Python y las librerías requeridas. Puedes hacerlo ejecutando:

```bash
pip install -r requirements.txt
```

### 3. Archivos de Entrada
Deberás colocar tu archivo CSV con los datos de entrada en la carpeta `data/input/` dentro del directorio donde se encuentra el proyecto. Asegúrate de que el archivo CSV esté correctamente formateado según las expectativas del modelo.

### 4. (Solo en Linux) Permisos de Ejecución
Si estás utilizando Linux, asegúrate de dar permisos de ejecución a los archivos de script con el siguiente comando:

```bash
chmod +x run_docker.sh
```

## Estructura del Proyecto

La estructura del proyecto debería verse de la siguiente manera:

```
/AA1-TUIA-Alsop-Hachen
  ├── docker/
      ├── Dockerfile
      ├── inferencia.py
      ├── requirements.txt
      ├── run_docker.bat
      ├── run_docker.sh
      ├── data/
          ├── input/
              └── input.csv  # Coloca tu archivo CSV aquí
          ├── output/
      ├── readme.md
```

## Construir la Imagen Docker

### En Windows

1. Navega al directorio `docker/` en la terminal.
2. Ejecuta el siguiente comando para construir la imagen de Docker:

```bash
build_docker_win.bat
```

### En Linux

1. Navega al directorio `docker/` en la terminal.
2. Ejecuta el siguiente comando para construir la imagen de Docker:

```bash
build_docker_linux.sh
```

Este comando construirá la imagen Docker con el nombre `tp2_alsop_hachen`, que contiene el modelo y sus dependencias.

## Ejecutar el Contenedor Docker

### En Windows

1. Asegúrate de tener tu archivo CSV dentro de la carpeta `data/input/`.
2. Ejecuta el archivo `.bat` con el siguiente comando en la terminal o haciendo doble clic sobre él:

```bash
run_docker_win.bat
```

El archivo `run_docker.bat` ejecutará el contenedor Docker y realizará la inferencia, guardando los resultados en la carpeta `data/output/`.

### En Linux

1. Asegúrate de tener tu archivo CSV dentro de la carpeta `data/input/`.
2. Ejecuta el script `.sh` con el siguiente comando en la terminal:

```bash
./run_docker_linux.sh
```

El archivo `run_docker.sh` ejecutará el contenedor Docker y realizará la inferencia, guardando los resultados en la carpeta `data/output/`.

## Datos de Entrada

Los datos de entrada deben ser proporcionados en formato CSV y deben colocarse dentro de la carpeta `data/input/`. El nombre del archivo CSV debe ser "input", pero debes asegurarte de que el archivo contenga las características que el modelo espera, el único valor obligatorio para que funcione correctamente la predicción es la fecha en formato AAAA-MM-DD.

**Ejemplo de estructura de archivo CSV:**

```
Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RainTomorrow
2008-12-01,Albury,13.4,22.9,0.6,NA,NA,W,44,W,WNW,20,24,71,22,1007.7,1007.1,8,NA,16.9,21.8,No,No
2008-12-02,Albury,7.4,25.1,0,NA,NA,WNW,44,NNW,WSW,4,22,44,25,1010.6,1007.8,NA,NA,17.2,24.3,No,No
```

## Datos de Salida

Una vez ejecutada la inferencia, los resultados se guardarán en la carpeta `data/output/` como un archivo generado por el modelo. El formato de salida es un archivo csv con las predicciones.

## Notas Adicionales

- Asegúrate de que Docker esté en ejecución antes de intentar ejecutar el contenedor.
- Si tienes problemas con las rutas o permisos, revisa que las carpetas `data/input` y `data/output` existan y tengan los permisos adecuados.
- El archivo `requirements.txt` contiene las librerías necesarias para ejecutar el modelo, en caso de que prefieras ejecutar la inferencia fuera de Docker.
- Si tienes alguna duda o problema, no dudes en abrir un *issue* en el repositorio.

---

## Construir la Imagen Docker sin los archivos de rápida ejecución

### En Windows

1. Navega al directorio `docker/` en la terminal.
2. Ejecuta el siguiente comando para construir la imagen de Docker:

```bash
docker build --no-cache -t tp2_alsop_hachen .
```

### En Linux

1. Navega al directorio `docker/` en la terminal.
2. Ejecuta el siguiente comando para construir la imagen de Docker:

```bash
docker build --no-cache -t tp2_alsop_hachen .
```

## Ejecutar el Contenedor Docker sin los archivos de rápida ejecución

### En Windows

1. Asegúrate de tener tu archivo CSV dentro de la carpeta `data/input/`.
2. Ejecuta el siguiente comando en la terminal o haciendo doble clic sobre él:

```bash
docker run --rm -v "{path donde tengas guardado el contenedor}\docker\data\input:/data/input" -v "{path donde tengas guardado el contenedor}\docker\data\output:/data/output" tp2_alsop_hachen
```

### En Linux

1. Asegúrate de tener tu archivo CSV dentro de la carpeta `data/input/`.
2. Ejecuta el script `.sh` con el siguiente comando en la terminal:

```bash
docker run --rm -v "{path donde tengas guardado el contenedor}/docker/data/input:/data/input" -v "{path donde tengas guardado el contenedor}/docker/data/output:/data/output" tp2_alsop_hachen
```