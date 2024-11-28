import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    'WindGustDir': ['N', 'E', 'W', 'SE', 'SW', 'NNE'],
    'WindDir9am': ['N', 'NE', 'E', 'SE', 'SSE', 'SW'],
    'WindDir3pm': ['NW', 'W', 'SW', 'E', 'N', 'W']
}
df = pd.DataFrame(data)
print(df)
# Mapeo de direcciones a ángulos
angle_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

# Asegurarse de que las columnas sean cadenas
df['WindGustDir'] = df['WindGustDir'].astype(str)
df['WindDir9am'] = df['WindDir9am'].astype(str)
df['WindDir3pm'] = df['WindDir3pm'].astype(str)

# Verificar si hay valores fuera del mapeo
for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
    print(f"Valores no encontrados en {col}:")
    print(df[~df[col].isin(angle_map.keys())])

# Mapear las direcciones a ángulos y convertir a float
for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
    df[f'{col}_angulo'] = df[col].map(angle_map).astype(float)

print(df)
