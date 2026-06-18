#type: ignore
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Garante que a pasta model existe para salvar o arquivo
os.makedirs("model", exist_ok=True)

# Define um modelo sequencial simples 
model = models.Sequential([
    layers.Input(shape=(28,28,1)),          
    layers.Conv2D(8, 3, activation="relu"), 
    layers.Flatten(),                       
    layers.Dense(1, activation="sigmoid")   
])

# Compila o modelo
model.compile(optimizer="adam", loss="binary_crossentropy")

# Salva o modelo em formato HDF5 na pasta 'model'
model.save("model/meu_modelo_teste.h5")

# Mensagem indicando onde o modelo foi salvo
print("Modelo de teste salvo em model\\meu_modelo_teste.h5")