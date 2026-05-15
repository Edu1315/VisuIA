import os
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=os.cpu_count())

def realizar_calculo_ia(arquivo):
    import time
    time.sleep(1) # Cada imagem "demora" 1 segundo
    import random
    return random.choice(["IA", "Autêntica"])

from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import random, datetime
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Carregar variáveis do .env
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# Configuração MongoDB Atlas
app.config["MONGO_URI"] = mongo_uri
mongo = PyMongo(app)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "ok", "uptime": "99%"})

@app.route('/analisar', methods=['POST'])
def analisar():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    resultado = random.choice(["Imagem gerada por IA", "Imagem autêntica", "Manipulação detectada"])
    confianca = random.randint(70, 99)

    analise = {
        "arquivo": file.filename,
        "resultado": resultado,
        "confianca": confianca,
        "data": datetime.datetime.now()
    }

    inserted = mongo.db.imagens.insert_one(analise)
    analise["id"] = str(inserted.inserted_id)
    analise.pop("_id", None)

    return jsonify(analise)

@app.route('/historico', methods=['GET'])
def get_historico():
    analises = []
    for a in mongo.db.imagens.find():
        a["id"] = str(a.pop("_id"))  # Converte o ID de objeto para texto
        analises.append(a)
    return jsonify(analises)

@app.route('/analisar_lote', methods=['POST'])
def analisar_lote():
    # Pega várias imagens enviadas de uma vez
    arquivos = request.files.getlist('imagens')
    resultados = list(executor.map(realizar_calculo_ia, arquivos))
    
    return jsonify({
        "status": "sucesso",
        "metodo": "Paralelismo (Multiprocessing)",
        "quantidade": len(resultados),
        "detalhes": resultados
    })

@app.route('/historico/<id>', methods=['GET'])
def get_analise(id):
    try:
        a = mongo.db.imagens.find_one({"_id": ObjectId(id)})
    except:
        return jsonify({"error": "ID inválido"}), 400

    if not a:
        return jsonify({"error": "Análise não encontrada"}), 404

    return jsonify({
        "id": str(a["_id"]),
        "arquivo": a["arquivo"],
        "resultado": a["resultado"],
        "confianca": a["confianca"],
        "data": a["data"]
    })

@app.route('/historico/<id>', methods=['DELETE'])
def delete_analise(id):
    try:
        result = mongo.db.imagens.delete_one({"_id": ObjectId(id)})
    except:
        return jsonify({"error": "ID inválido"}), 400

    if result.deleted_count == 0:
        return jsonify({"error": "Análise não encontrada"}), 404

    return jsonify({"message": "Análise removida com sucesso"})

if __name__ == '__main__':
    from waitress import serve
    print("Servidor rodando com Waitress em http://127.0.0.1:5000")
    serve(app, host='127.0.0.1', port=5000)