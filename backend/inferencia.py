# backend/inferencia.py
import numpy as np
import threading

_model = None
_model_lock = threading.Lock()

def get_model(loader):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = loader()  # loader deve retornar o modelo carregado
    return _model

def normalize_prob(x):
    x = float(x)
    return max(0.0, min(1.0, x))

def inferir_imagem(model, image_array):
    """
    Recebe array numpy pré-processado e retorna dicionário padronizado:
    { "ai_prob": 0..1, "confidence": 0..1, "raw": {...} }
    Ajuste a extração de prob conforme a saída real do seu modelo.
    """
    # exemplo genérico: model.predict retorna prob de classe IA na posição [0][0]
    preds = model.predict(np.expand_dims(image_array, axis=0))
    # ajuste aqui se seu modelo retornar logits ou vetor de classes
    ai_prob = float(preds[0][0])
    confidence = ai_prob  # ou calcule outra métrica
    return {
        "ai_prob": normalize_prob(ai_prob),
        "confidence": normalize_prob(confidence),
        "raw": {"model_output": preds.tolist()}
    }
