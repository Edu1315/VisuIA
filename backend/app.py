
import os
import uuid
import traceback
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
from database.mongo import mongo, init_db
from bson.objectid import ObjectId
import datetime
from dotenv import load_dotenv

# Redução de logs do TensorFlow: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Desliga otimizações oneDNN (Reduz incompatibilidade)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from PIL import Image
import numpy as np

from transformers import pipeline

# Inicializando Hugging Face detector 
try:
    hf_detector1 = pipeline("image-classification", model="capcheck/ai-image-detection")
    hf_detector2 = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    print("HF detector carregado com sucesso")
except Exception as e:
    hf_detector = None
    print("Falha ao carregar HF detector:", e)

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)
init_db(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #Tamanho maximo 16mb

# Pasta de uploads temporários
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Carregar modelo local caso HF esteja indisponivel
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "mobilenet_v2.h5"))
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso:", MODEL_PATH)
except Exception as e:
    model = None
    print("Falha ao carregar modelo local:", e)

# Executor de threads (Paralelismo)
MAX_WORKERS = 4
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Formata o documento do mongo para o front
def serialize_for_front(doc):
    out = {}
    out["id"] = str(doc.get("_id") or doc.get("id") or "")
    out["arquivo"] = doc.get("filename") or doc.get("arquivo") or ""

    # Resultado textual (compatibilidade)
    if "resultado" in doc and doc.get("resultado") is not None:
        out["resultado"] = doc.get("resultado")
    else:
        r = doc.get("result", {}) if isinstance(doc.get("result", {}), dict) else {}
        ai = r.get("ai_prob")
        if ai is None:
            out["resultado"] = "Indeterminado"
        else:
            pct = round(float(ai) * 100, 2)
            out["resultado"] = f"A imagem tem {pct}% de chance de ser IA"


    # Confianca numérica (0..100) para exibição
    if "confianca" in doc and doc.get("confianca") is not None:
        try:
            out["confianca"] = float(doc.get("confianca"))
        except Exception:
            out["confianca"] = None
    else:
        r = doc.get("result", {}) if isinstance(doc.get("result", {}), dict) else {}
        conf = r.get("confidence")
        out["confianca"] = round(float(conf) * 100, 2) if conf is not None else None

    # data em ISO legível (data de dia não de dados)
    created = doc.get("created_at") or doc.get("data")
    if hasattr(created, "isoformat"):
        out["data"] = created.isoformat()
    else:
        out["data"] = created

  
    out["result"] = doc.get("result")
    out["error"] = doc.get("error")
    return out

# Recebe o caminho do arquivo e retorna padronizado
def inferir_caminho(caminho):
    """
    Retorna dicionário padronizado:
    { "arquivo": nome, "result": {"ai_prob":0..1 or None,"confidence":0..1 or None,"raw":{...}}, "erro": None ou mensagem }
    """
    import os
    import traceback
    from PIL import Image
    import numpy as np

    # Extrai o nome original da imagem (retira o ID do resultado no front)
    nome_original = None
    try:

        try:
            nome_original = os.path.basename(caminho).split("_", 1)[1]
        except Exception:
            nome_original = os.path.basename(caminho) or "arquivo-desconhecido"

        # Pré-processa a imagem 
        img = Image.open(caminho).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        inp = np.expand_dims(arr, axis=0)

        # Variaveis de Saida
        ai_prob = None
        confidence = None
        raw_out = None

        # Detectores
        try:
            if hf_detector1 and hf_detector2:
                res1 = hf_detector1(caminho)
                res2 = hf_detector2(caminho)
                print("DEBUG: res1:", res1)
                print("DEBUG: res2:", res2)

                # Seleciona a predição com maior score de cada detector
                top1 = max(res1, key=lambda x: x["score"])
                top2 = max(res2, key=lambda x: x["score"])

                label1, score1 = top1["label"].lower(), float(top1["score"])
                label2, score2 = top2["label"].lower(), float(top2["score"])

                # Decisão ensemble (concordância entre detectores define ai_prob)
                if ("fake" in label1 or "artificial" in label1) and ("fake" in label2 or "artificial" in label2):
                    ai_prob = max(score1, score2)
                    confidence = max(score1, score2)
                elif ("real" in label1 or "human" in label1) and ("real" in label2 or "human" in label2):
                    ai_prob = min(1.0 - score1, 1.0 - score2)
                    confidence = max(score1, score2)
                else:
                    ai_prob = None
                    confidence = None

                raw_out = {"hf_result1": res1, "hf_result2": res2}
        except Exception as e:
            traceback.print_exc()
            arquivo_fallback = nome_original or (os.path.basename(caminho) if caminho else "arquivo-desconhecido")
            return {
                "aexcept Exception rquivo": arquivo_fallback,
                "resultado": "Indeterminado",
                "result": {"ai_prob": None, "confidence": None, "raw": None},
                "erro": str(e)
            }


        # Garante limites para ai_prob e confidence
        if ai_prob is not None:
            ai_prob = max(0.0, min(1.0, float(ai_prob)))
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))

        # Interpretação dos resultados
        if ai_prob is None:
            resultado_texto = "Indeterminado"
        else:
            pct = round(ai_prob * 100, 2)
            if ai_prob >= 0.80:
                resultado_texto = f"A imagem tem {pct}% de chance de ser IA (provavelmente IA)"
            elif ai_prob <= 0.20:
                resultado_texto = f"A imagem tem {pct}% de chance de ser IA (provavelmente real)"
            else:
                resultado_texto = "Indeterminado"

        return {
            "arquivo": nome_original,
            "resultado": resultado_texto,
            "result": {"ai_prob": ai_prob, "confidence": confidence, "raw": raw_out},
            "erro": None
        }

    except Exception as e:
        traceback.print_exc()
        arquivo_fallback = nome_original or (os.path.basename(caminho) if caminho else "arquivo-desconhecido")
        return {
            "arquivo": arquivo_fallback,
            "resultado": "Indeterminado",
            "result": {"ai_prob": None, "confidence": None, "raw": None},
            "erro": str(e)
        }



# Checar a "saude" do serviço
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "ok", "uptime": "99%"})

# Rota para analisar uma única imagem
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

        # calcular probabilidade de ser IA (0..1) a partir de ai_prob
        ai_prob = result.get("ai_prob")
        chance_ia_pct = None if ai_prob is None else round(float(ai_prob) * 100, 2)

        # documento salvo no Mongo
        doc = {
            "filename": infer.get("arquivo", file.filename),
            "arquivo": infer.get("arquivo", file.filename),   
            # armazenar a probabilidade de ser IA 
            "resultado": f"A imagem tem {chance_ia_pct}% de chance de ser IA" if chance_ia_pct is not None else None,
            "result": result,
            "error": erro,
            "created_at": datetime.datetime.utcnow()
        }

        inserted = mongo.db.imagens.insert_one(doc)  # *coleção: imagens*
        doc["_id"] = inserted.inserted_id

        resp = serialize_for_front(doc)
        return jsonify(resp), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    # Remove arquivo temporário após processamento
    finally:
        try:
            if os.path.exists(caminho):
                os.remove(caminho)
        except Exception:
            pass



# Rota para analisar múltiplas imagens em lote
@app.route('/analisar_lote', methods=['POST'])
def analisar_lote():
    arquivos = request.files.getlist('imagens')
    if not arquivos:
        return jsonify({"status": "erro", "mensagem": "Nenhuma imagem enviada"}), 400
    
    # Salva cada upload temporariamente e registra caminho original
    caminhos = []
    for arq in arquivos:
        nome_unico = f"{uuid.uuid4().hex}_{arq.filename}"
        caminho = os.path.join(UPLOAD_DIR, nome_unico)
        arq.save(caminho)
        caminhos.append((caminho, arq.filename))

    
    future_to_name = {}
    for caminho, original_name in caminhos:
        # Submete cada arquivo para o executor processar de forma paralela
        f = executor.submit(inferir_caminho, caminho)
        future_to_name[f] = (caminho, original_name)

    resultados = []
    try:
        # Coleta resultados conforme as threads terminam
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
            chance_ia_pct = None if ai_prob is None else round(float(ai_prob) * 100, 2)
            confidence = result.get("confidence")
            confidence_pct = None if confidence is None else round(float(confidence) * 100, 2)

            doc = {
                "filename": res.get("arquivo", original_name),
                "arquivo": res.get("arquivo", original_name),
                "resultado": f"A imagem tem {chance_ia_pct}% de chance de ser IA" if chance_ia_pct is not None else None,
                "confianca": confidence_pct,
                "result": result,
                "error": res.get("erro"),
                # salvar created_at com timezone UTC
                "created_at": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            }

            # Insere cada análise no Mongo
            inserted = mongo.db.imagens.insert_one(doc)
            doc["_id"] = inserted.inserted_id
            resultados.append(serialize_for_front(doc))

    except Exception:
        resultados.append({"filename": None, "result": {"ai_prob": None, "confidence": None}, "error": traceback.format_exc()})
    finally:
        # limpa todos os arquivos temporários
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
        # Retorna histórico de análises (últimos 100)
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