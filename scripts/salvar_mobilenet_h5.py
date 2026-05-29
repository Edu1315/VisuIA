# salvar_mobilenet_h5.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import os

os.makedirs("model", exist_ok=True)
model = MobileNetV2(weights="imagenet", include_top=True)
model.save("model/mobilenet_v2.h5")   # salva como HDF5
print("Arquivo .h5 salvo em model/mobilenet_v2.h5")
