from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app, resources={r"/prever": {"origins": "*"}}, supports_credentials=True)

# Variáveis globais para lazy-loading
model = None
classes = None
scaler = None
colunas_treinadas = None
df_paises = None

# Carregamento só na primeira requisição
def carregar_tudo():
    global model, classes, scaler, colunas_treinadas, df_paises
    if model is None:
        model = tf.keras.models.load_model("modelo_perfis.keras")
        classes = np.load("encoder_perfis.npy", allow_pickle=True)
        scaler = joblib.load("scaler_perfis.pkl")
        colunas_treinadas = joblib.load("colunas_treinadas.pkl")
        df_paises = pd.read_csv("base_paises_perfis.csv", low_memory=True)

# Mapeamento entre modelo e nome legível
mapa_perfis = {
    "Sol e Praia": "Sol e Praia",
    "Cultura e História": "Cultura e História",
    "Aventura e Natureza": "Aventura e Natureza",
    "Frio e Montanha": "Frio e Montanha"
}

@app.route('/')
def home():
    return jsonify({"mensagem": "API de recomendação de destinos está online."})

@app.route('/prever', methods=['POST'])
def prever():
    carregar_tudo()

    dados = request.json
    if not dados or 'binarios' not in dados or 'preferencias' not in dados:
        return jsonify({'erro': 'Dados incompletos'}), 400

    binarios = dados['binarios']
    preferencias = dados['preferencias']

    df_usuario = pd.DataFrame([binarios])
    df_usuario = df_usuario.reindex(columns=colunas_treinadas, fill_value=0)
    X_usuario = scaler.transform(df_usuario)

    pred = model.predict(X_usuario)
    indice_predito = np.argmax(pred)
    perfil_previsto = classes[indice_predito]
    perfil_csv = mapa_perfis.get(perfil_previsto)

    if not perfil_csv:
        return jsonify({'erro': f"Perfil '{perfil_previsto}' não tem correspondência."}), 500

    prefs = {
        "idioma": preferencias["idioma"].lower().strip(),
        "continente": preferencias["continente"].lower().strip(),
        "cultura": preferencias["cultura"].lower().strip(),
        "orcamento": preferencias["orcamento"].replace("_", "-").lower().strip()
    }

    candidatos = df_paises[df_paises["perfil"].str.lower() == perfil_csv.lower()].copy()

    if candidatos.empty:
        return jsonify({
            'perfil': perfil_previsto,
            'destinos': [],
            'mensagem': "Nenhum país com esse perfil."
        })

    def pontuar(linha):
        score = 0
        if prefs["continente"] in linha["continente"].lower():
            score += 4
        if prefs["orcamento"] in linha["orcamento"].lower():
            score += 3
        if prefs["idioma"] in linha["idioma"].lower():
            score += 2
        if prefs["cultura"] in linha["cultura"].lower():
            score += 1
        return score

    candidatos["score"] = candidatos.apply(pontuar, axis=1)
    candidatos = candidatos.sort_values(by="score", ascending=False)

    if candidatos["score"].max() == 0:
        return jsonify({
            'perfil': perfil_previsto,
            'destinos': [],
            'mensagem': "Nenhum país compatível com as preferências."
        })

    top3 = candidatos.head(3)
    nomes_top3 = top3["nome"].tolist()

    return jsonify({
        'perfil': perfil_previsto,
        'destinos': nomes_top3,
        'mensagem': f"Baseado no perfil '{perfil_previsto}', recomendamos estes destinos."
    })

if __name__ == '__main__':
    app.run()
