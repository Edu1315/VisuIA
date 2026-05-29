import os
import uuid
import traceback
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
from database.mongo import mongo, init_db
from bson.objectid import ObjectId
import datetime
from dotenv import load_dotenv

import os
# reduzir logs do TensorFlow: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# desliga otimizações oneDNN se quiser resultados numéricos consistentes
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf



from PIL import Image
import numpy as np
import tensorflow as tf 


load_dotenv()

app = Flask(__name__)
init_db(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# pasta de uploads temporários
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# carregar modelo (caminho absoluto e checagem)
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "mobilenet_v2.h5"))
model = tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso:", MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

# executor de threads
MAX_WORKERS = 4
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def serialize_for_front(doc):
    """
    Retorna dicionário com os campos que o front espera:
    id, arquivo, resultado, confianca, data
    Mantém também 'result' (novo formato) e 'error' para debug.
    """
    out = {}
    out["id"] = str(doc.get("_id") or doc.get("id") or "")
    out["arquivo"] = doc.get("filename") or doc.get("arquivo") or ""
    # resultado textual (compatibilidade): prioriza label antigo, senão usa ai_prob interpretado
    if "resultado" in doc and doc.get("resultado") is not None:
        out["resultado"] = doc.get("resultado")
    else:
        # tenta derivar de result.ai_prob
        r = doc.get("result", {})
        ai = r.get("ai_prob") if isinstance(r, dict) else None
        if ai is None:
            out["resultado"] = "—"
        else:
            out["resultado"] = "Imagem gerada por IA" if ai >= 0.5 else "Imagem autêntica"
    # confianca numérica (0..100) — prioriza campo legado 'confianca' se existir
    if "confianca" in doc and doc.get("confianca") is not None:
        try:
            out["confianca"] = float(doc.get("confianca"))
        except Exception:
            out["confianca"] = None
    else:
        r = doc.get("result", {})
        conf = r.get("confidence") if isinstance(r, dict) else None
        out["confianca"] = round(float(conf) * 100, 2) if conf is not None else None
    # data em ISO legível
    created = doc.get("created_at") or doc.get("data")
    if hasattr(created, "isoformat"):
        out["data"] = created.isoformat()
    else:
        out["data"] = created
    # manter payload novo para frontend avançado
    out["result"] = doc.get("result")
    out["error"] = doc.get("error")
    return out


# substitua a função inferir_caminho por esta
def inferir_caminho(caminho):
    """
    Retorna dicionário padronizado:
    { "arquivo": nome, "result": {"ai_prob":0..1,"confidence":0..1,"raw":{...}}, "erro": None }
    """
    try:
        img = Image.open(caminho).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        # não expanda aqui se a função inferir_imagem espera shape (224,224,3)
        inp = np.expand_dims(arr, axis=0)
        preds = model.predict(inp)

        # ajuste conforme saída do seu modelo:
        # caso binário com saída shape (1,1)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            ai_prob = float(preds[0][0])
            confidence = ai_prob
        else:
            # se for vetor de probabilidades, tente assumir classe 0 = IA
            try:
                ai_prob = float(preds[0][0])
                confidence = max(preds[0]).item()
            except Exception:
                ai_prob = 0.0
                confidence = 0.0

        # garantir limites
        ai_prob = max(0.0, min(1.0, ai_prob))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "arquivo": os.path.basename(caminho),
            "result": {"ai_prob": ai_prob, "confidence": confidence, "raw": {"model_output": preds.tolist()}},
            "erro": None
        }
    except Exception as e:
        return {
            "arquivo": os.path.basename(caminho),
            "result": {"ai_prob": None, "confidence": None, "raw": None},
            "erro": str(e)
        }


@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "ok", "uptime": "99%"})

@app.route('/analisar', methods=['POST'])
def analisar():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    nome_unico = f"{uuid.uuid4().hex}_{file.filename}"
    caminho = os.path.join(UPLOAD_DIR, nome_unico)
    file.save(caminho)

    try:
        infer = inferir_caminho(caminho)  # retorna {"arquivo", "result": {"ai_prob","confidence","raw"}, "erro"}
        result = infer.get("result", {"ai_prob": None, "confidence": None, "raw": None})
        erro = infer.get("erro")

        # calcular probabilidade de autenticidade (0..1) a partir de ai_prob
        ai_prob = result.get("ai_prob")
        auth_prob = None if ai_prob is None else max(0.0, min(1.0, 1.0 - float(ai_prob)))
        confidence = result.get("confidence")
        confidence_pct = None if confidence is None else round(float(confidence) * 100, 2)

        # documento salvo no Mongo: inclui campos legados e novo 'result'
        doc = {
            "filename": infer.get("arquivo", file.filename),
            "arquivo": infer.get("arquivo", file.filename),   # legado
            # armazenar a probabilidade de autenticidade como string legada
            "resultado": f"{round(auth_prob * 100, 2)}%" if auth_prob is not None else None,
            "confianca": confidence_pct,
            "result": result,
            "error": erro,
            "created_at": datetime.datetime.utcnow()
        }
        inserted = mongo.db.imagens.insert_one(doc)  # **coleção: imagens**
        doc["_id"] = inserted.inserted_id

        resp = serialize_for_front(doc)
        return jsonify(resp), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(caminho):
                os.remove(caminho)
        except Exception:
            pass




@app.route('/analisar_lote', methods=['POST'])
def analisar_lote():
    arquivos = request.files.getlist('imagens')
    if not arquivos:
        return jsonify({"status": "erro", "mensagem": "Nenhuma imagem enviada"}), 400

    caminhos = []
    for arq in arquivos:
        nome_unico = f"{uuid.uuid4().hex}_{arq.filename}"
        caminho = os.path.join(UPLOAD_DIR, nome_unico)
        arq.save(caminho)
        caminhos.append((caminho, arq.filename))

    # submeter tarefas e mapear future -> nome original
    future_to_name = {}
    for caminho, original_name in caminhos:
        f = executor.submit(inferir_caminho, caminho)
        future_to_name[f] = (caminho, original_name)

    resultados = []
    try:
        for f in as_completed(list(future_to_name.keys())):
            caminho, original_name = future_to_name[f]
            try:
                res = f.result()
            except Exception as e:
                # erro na inferência dessa imagem
                resultados.append({
                    "filename": original_name,
                    "arquivo": original_name,
                    "resultado": None,
                    "confianca": None,
                    "result": {"ai_prob": None, "confidence": None},
                    "error": str(e),
                    "created_at": datetime.datetime.utcnow().isoformat()
                })
                continue

            result = res.get("result", {"ai_prob": None, "confidence": None, "raw": None})
            ai_prob = result.get("ai_prob")
            auth_prob = None if ai_prob is None else max(0.0, min(1.0, 1.0 - float(ai_prob)))
            confidence = result.get("confidence")
            confidence_pct = None if confidence is None else round(float(confidence) * 100, 2)

            doc = {
                "filename": res.get("arquivo", original_name),
                "arquivo": res.get("arquivo", original_name),
                "resultado": f"{round(auth_prob * 100, 2)}%" if auth_prob is not None else None,
                "confianca": confidence_pct,
                "result": result,
                "error": res.get("erro"),
                "created_at": datetime.datetime.utcnow()
            }
            inserted = mongo.db.imagens.insert_one(doc)
            doc["_id"] = inserted.inserted_id
            resultados.append(serialize_for_front(doc))

    except Exception:
        resultados.append({"filename": None, "result": {"ai_prob": None, "confidence": None}, "error": traceback.format_exc()})
    finally:
        # limpar arquivos temporários
        for c, _ in caminhos:
            try:
                os.remove(c)
            except Exception:
                pass

    return jsonify({"status": "sucesso", "quantidade": len(resultados), "detalhes": resultados}), 200




@app.route('/historico', methods=['GET'])
def get_historico():
    try:
        docs = []
        for a in mongo.db.imagens.find().sort("created_at", -1).limit(100):
            docs.append(serialize_for_front(a))
        return jsonify(docs), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/historico/<id>', methods=['GET'])
def get_analise(id):
    try:
        a = mongo.db.imagens.find_one({"_id": ObjectId(id)})
    except Exception:
        return jsonify({"error": "ID inválido"}), 400
    if not a:
        return jsonify({"error": "Análise não encontrada"}), 404
    return jsonify(serialize_for_front(a)), 200

@app.route('/historico/<id>', methods=['DELETE'])
def delete_analise(id):
    try:
        result = mongo.db.imagens.delete_one({"_id": ObjectId(id)})
    except Exception:
        return jsonify({"error": "ID inválido"}), 400
    if result.deleted_count == 0:
        return jsonify({"error": "Análise não encontrada"}), 404
    return jsonify({"message": "Análise removida com sucesso"}), 200



if __name__ == '__main__':
    from waitress import serve
    print("Servidor rodando com Waitress em http://127.0.0.1:5000")
    serve(app, host='127.0.0.1', port=5000)
