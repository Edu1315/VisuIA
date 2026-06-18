#type: ignore
# Script simples para baixar o MobileNetV2 pré-treinado e salvar como arquivo HDF5
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import os

# Garante que a pasta model exista para salvar o arquivo
os.makedirs("model", exist_ok=True)

# Carrega MobileNetV2 com pesos pré-treinados no ImageNet
# - weights="imagenet": usa pesos treinados no ImageNet
model = MobileNetV2(weights="imagenet", include_top=True)

# Salva o modelo no formato HDF5 (.h5) dentro da pasta model
model.save("model/mobilenet_v2.h5")  

# Mensagem informando onde o arquivo foi salvo
print("Arquivo .h5 salvo em model/mobilenet_v2.h5")