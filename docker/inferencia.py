from pycaret.classification import load_model
import pandas as pd
import sys
from pycaret.classification import load_model
from pycaret.classification import predict_model
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import RobustScaler

import joblib

import sys

def process_input(input_data):
    processed_data = []
    for item in input_data:
        try:
            # Intentar convertir a float
            processed_data.append(float(item))
        except ValueError:
            # Si no es numérico, agregar como string
            processed_data.append(item)
    
    return processed_data

class NeuralNetwork:
    def __init__(self, epochs=50, batch_size=16, learning_rate=0.01):
        #inicializo algunos parámetros como épocas, batch_size, learning rate
        #(no son necesarios)
        #se puede agregar la cantidad de capas, la cantidad de neuronas por capa (pensando en hacer una clase que pueda ser usada para cualquier caso)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self, input_shape):
        # Definir el modelo con 3 capas ocultas
        model = tf.keras.models.Sequential([
            # Capa densa con regularización L2
            tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),  # Dropout para evitar sobreajuste
            tf.keras.layers.Dense(26, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            # Capa de salida con sigmoide para clasificación binaria
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Ajustar la tasa de aprendizaje
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Reducir la tasa de aprendizaje si es necesario
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # Función de pérdida binaria

        self.model = model

    def fit(self, X_train, y_train, X_valid, y_valid):
        # simplemente el fit del modelo. Devuelvo la evolución de la función de pérdida, ya que es interesante ver como varía a medida que aumentan las épocas!
        history=self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=self.epochs, batch_size=self.batch_size)
        return history.history['loss'], history.history['val_loss']

    def evaluate(self, X_test, y_test):
        ### evalúo en test
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"test accuracy: {accuracy:.4f}")

    def predict(self, X_new):
        ### predicciones
        predictions = self.model.predict(X_new)
        predicted_classes = (predictions > 0.5).astype(int)
        return predicted_classes

# def predict(input_data):
#     # Definir las columnas en el orden correcto
#     columns = [
#         "Date", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", 
#         "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", 
#         "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", 
#         "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", 
#         "Temp9am", "Temp3pm", "RainToday"
#     ]

#     # Crear un DataFrame con los datos de entrada
#     df = pd.DataFrame([input_data], columns=columns)

#     # Convertir tipos si es necesario
#     df["Date"] = pd.to_datetime(df["Date"])  # Asegurar que la fecha esté en formato datetime
#     categorical_columns = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
#     for col in categorical_columns:
#         df[col] = df[col].astype("category")

#     # Realizar la predicción
#     prediction = pipeline.predict(df.drop(columns=["Date"]))  # Ignorar 'Date' si fue ignorada en el entrenamiento
#     return "Sí lloverá mañana" if prediction[0] == 1 else "No lloverá mañana"


if __name__ == "__main__":
    # Leer los datos de entrada desde la línea de comando
    nn = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    input_data = sys.argv[1].split(",")
    input_data = process_input(input_data)
    print(input_data)
    columns = [
        "Date", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", 
        "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", 
        "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", 
        "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", 
        "Temp9am", "Temp3pm", "RainToday"
    ]
    
    df = pd.DataFrame([input_data], columns=columns)

    category_variable = ['WindGustDir','WindDir9am','WindDir3pm']

    for var in category_variable:
        df[var] = df[var].astype(str)

    df['Date'] = df['Date'].astype('datetime64[ns]')

    #print(df)
    # Mapeamos las direcciones del viento a ángulos
    angle_map = {
        ' N': 0, ' NNE': 22.5, ' NE': 45, ' ENE': 67.5,
        ' E': 90, ' ESE': 112.5, ' SE': 135, ' SSE': 157.5,
        ' S': 180, ' SSW': 202.5, ' SW': 225, ' WSW': 247.5,
        ' W': 270, ' WNW': 292.5, ' NW': 315, ' NNW': 337.5
    }

    # Se mapea cada dirección a su ángulo
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        df[f'{col}_angulo'] = df[col].map(angle_map).astype(float) # Se debe transformar a float, ya que si no serán categoricas
    # Convertimos los ángulos a valores seno y coseno
    for col in ['WindGustDir_angulo', 'WindDir9am_angulo', 'WindDir3pm_angulo']:
        df[f'{col}_sin'] = np.sin(np.radians(df[col])) # Se pasan a radianes ya que np.sin y np.cos esperan ángulos en radianes
        df[f'{col}_cos'] = np.cos(np.radians(df[col]))
    
    # Descartamos las columnas originales y las _angulo
    x_train_mapping = df.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm',
                                'WindGustDir_angulo', 'WindDir9am_angulo', 'WindDir3pm_angulo'])
    
    x_train_mapping['Month'] = x_train_mapping['Date'].dt.month
    x_train_mapping['Month_sin'] = np.sin(2 * np.pi * x_train_mapping['Month'] / 12)
    x_train_mapping['Month_cos'] = np.cos(2 * np.pi * x_train_mapping['Month'] / 12)

    x_train_mapping = x_train_mapping.drop(columns=['Month'])
    
    #print(df)
    from sklearn.preprocessing import LabelEncoder

    # Hacemos una copia para trabajar en ella
    x_train_imputer_v2 = x_train_mapping.copy()

    for var in ['RainToday']:
        label_encoder = LabelEncoder()
        
        non_null_train = x_train_imputer_v2[var].notnull()

        
        # Ajusta el codificador en los datos no nulos
        x_train_imputer_v2.loc[non_null_train, var] = label_encoder.fit_transform(x_train_imputer_v2.loc[non_null_train, var])
    
    x_train_imputer_v2 = x_train_imputer_v2.drop(columns=['Date']).reset_index(drop=True)
    
    #scaler = RobustScaler()
    x_train_scaled = scaler.transform(x_train_imputer_v2)

    print(x_train_scaled)

    pred = nn.predict(x_train_scaled)

    print("Predicción:", pred)

