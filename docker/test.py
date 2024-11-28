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

#data = pd.read_csv('weatherAUS.csv')


# Cargar el pipeline finalizado
#pipeline = load_model("pipeline")
import joblib
#nn = load_model("model")
nn = joblib.load("model.pkl")
# Datos originales de entrada
input_data = [ 0.25773196,  0.44274809,  0.        ,  0.52380952,  0.36111111,
       -0.4       , -0.33333333,  0.        , -0.56666667, -0.76666667,
        0.        , -0.16091954,  0.2       ,  0.22727273,  0.66363636,
        0.48780488,  0.        ,  0.65328148,  0.27059805,  0.27059805,
       -0.65328148,  0.5       ,  0.5       ,  0.5       , -0.6339746 ]
columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
       'Temp9am', 'Temp3pm', 'RainToday', 'WindGustDir_angulo_sin',
       'WindGustDir_angulo_cos', 'WindDir9am_angulo_sin',
       'WindDir9am_angulo_cos', 'WindDir3pm_angulo_sin',
       'WindDir3pm_angulo_cos', 'Month_sin', 'Month_cos']

# Crear un DataFrame
df = pd.DataFrame([input_data], columns=columns)
#df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# Realizar la predicción
#prediction = predict_model(pipeline, data=df)
print(df)
pred = nn.predict(df)
print("Predicción:", pred)

