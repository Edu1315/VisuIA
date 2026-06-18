import numpy as np
import threading

# Instância global do modelo e lock para carregamento
_model = None
_model_lock = threading.Lock()

# Retorna modelo carregado. Usa loader para carregar só uma vez
def get_model(loader):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = loader()  
    return _model

# Normaliza probabilidade para o intervalo 0.0, 1.0
def normalize_prob(x):
    x = float(x)
    return max(0.0, min(1.0, x))

# Executa inferência e retorna ai_prob e confidence
def inferir_imagem(model, image_array):
    """
    Recebe array numpy pré-processado e retorna dicionário padronizado:
    { "ai_prob": 0..1, "confidence": 0..1, "raw": {...} }
    Ajuste a extração de prob conforme a saída real do seu modelo.
    """
    preds = model.predict(np.expand_dims(image_array, axis=0))
    ai_prob = float(preds[0][0])
    confidence = ai_prob 

    # Retorna dicionário padronizado
    return {
        "ai_prob": normalize_prob(ai_prob),
        "confidence": normalize_prob(confidence),
        "raw": {"model_output": preds.tolist()}
    }