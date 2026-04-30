from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import random, datetime

app = Flask(__name__)

# Configuração MongoDB (ajuste se usar Atlas)
app.config["MONGO_URI"] = "mongodb://localhost:27017/visuia"
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
        "data": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    }

    inserted = mongo.db.analises.insert_one(analise)
    analise["id"] = str(inserted.inserted_id)

    return jsonify(analise)

@app.route('/historico', methods=['GET'])
def get_historico():
    analises = []
    for a in mongo.db.analises.find():
        analises.append({
            "id": str(a["_id"]),
            "arquivo": a["arquivo"],
            "resultado": a["resultado"],
            "confianca": a["confianca"],
            "data": a["data"]
        })
    return jsonify(analises)

@app.route('/historico/<id>', methods=['GET'])
def get_analise(id):
    try:
        a = mongo.db.analises.find_one({"_id": ObjectId(id)})
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
        result = mongo.db.analises.delete_one({"_id": ObjectId(id)})
    except:
        return jsonify({"error": "ID inválido"}), 400

    if result.deleted_count == 0:
        return jsonify({"error": "Análise não encontrada"}), 404

    return jsonify({"message": "Análise removida com sucesso"})

if __name__ == '__main__':
    app.run(debug=True)