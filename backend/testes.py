# salvar_modelo_teste.py
import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.makedirs("model", exist_ok=True)

model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(8,3,activation="relu"),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.save("model/meu_modelo_teste.h5")
print("Modelo de teste salvo em model\\meu_modelo_teste.h5")
