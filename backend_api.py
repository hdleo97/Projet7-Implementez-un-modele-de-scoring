# pip install flask
# from flask import flask, jsonify, request
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans
#
# app = Flask(__name__)
#
# @app.route("/", methods=["GET"])
#
# @app.before_first_request
#
# def load_data():
#     data = pd.read_csv('application_test.csv', index_col='SK_ID_CURR', encoding ='utf-8')
#     data_train = pd.read_csv('application_train.csv', index_col='SK_ID_CURR', encoding ='utf-8')
#     target = data['TARGET']
#     return data, data_train, target
#
# def load():
#     model_path = "mlflow_model/model"
#     model = load_model(model_path, compile = False)
#     return model
#
# def load_age_population(data):
#     data_age = round((data["DAYS_BIRTH"]/365), 2)
#     return data_age
#
# def load_income_population(sample):
#     df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
#     df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
#     return df_income
# #Chargement du model
# model = load()
#
# def identite_client(data, id):
#     data_client = data[data.index == int(id)]
#     return data_client
#
# def load_prediction(sample, id, clf):
#     X=sample.iloc[:, :-1]
#     score = clf.predict_proba(X[X.index == int(id)])[:,1]
#     return score
#
# #Loading data……
#     data, data_train, target = load_data()
#     id_client = data.index.values
#     model = load()
#
# if __name__ = "__main__":
#     app.run(host = "0.0.0.0", port = 8080)

# Importation des bibliothèques
from fastapi import FastAPI, Path
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from tensorflow.keras.models import load_model
import numpy as np
import io

# Initialisation d'une instance de l'API
app = FastAPI()

# Chargement du modèle de prédiction de crédit
model = joblib.load("mlflow_model/model.pkl")

# Chargement du dataframe de données
data = pd.read_csv('application_train.csv', encoding ='utf-8')
data_test = pd.read_csv('application_test.csv', encoding ='utf-8')
target = data['TARGET']

# Définition des caractéristiques pertinentes (isolement des features non utilisées)
# ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
# relevant_features = [col for col in df.columns if col not in ignore_features]

# Création de l'explainer shap
explainer = shap.LinearExplainer(model)

# Création d'une classe Pydantic pour les paramètres d'entrée
class ClientRequest(BaseModel):
    id_client: int

# Définition d'une route vers la racine de l'API
@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI API!"}

# Définition de la route pour obtenir la liste des identifiants clients
@app.get("/client_ids")
async def get_client_ids():
    clients_ids =  data['SK_ID_CURR'].tolist()
    return clients_ids

# Défintion d'une route pour obtenir la liste des features
@app.get("/features")
async def get_features():
    features = relevant_features
    return features

# Définition de la route de prédiction de crédit ("/credit/{id_client}")
@app.get("/credit/{id_client}")
async def predict_credit(id_client: int = Path(..., title="Client ID")):
    # Filtre sur les données du client en question
    X = df[data['SK_ID_CURR'] == id_client]
    X = X[relevant_features]

    # Calcul de la prédiction et de la probabilité de prédiction
    proba = load_clf.predict_proba(X)
    prediction = load_clf.predict(X)

    # Afficher la probabilité avec 2 chiffres après la virgule
    proba_formatted = round(float(proba[0][0]), 2)

    # Interprétation
    interpretation = ""
    if prediction[0] == 0:
        interpretation = f"Client solvable avec une probabilité égale à {proba_formatted}"
    else:
        interpretation = f"Client non solvable avec une probabilité égale à {proba_formatted}"

    # Création de la réponse de la prédiction
    pred_proba = {
        'Prédiction': int(prediction[0]),
        'Probabilité': proba_formatted,
        'Conclusion': interpretation
    }

    # Retour de la réponse de la prédiction
    return pred_proba

# Définition de la route pour récupérer les données d'un client spécifique
@app.get("/client_data/{id_client}")
async def get_client_data(id_client: int = Path(..., title="Client ID")):
    # Filtrer les données du client en question
    client_data = data[data['SK_ID_CURR'] == id_client]

    return client_data.to_dict(orient="records")

# Définition de la route pour récupérer les données de 1000 clients choisis de manière aléatoire
@app.get("/all_clients_data")
async def get_all_clients_data():

   # Échantillon aléatoire de 1000 clients
   random_clients = data.sample(n=1000, random_state=42)

   return random_clients.to_dict(orient="records")

# Définition de la route pour calculer les plus proches voisins du client_id
@app.get("/nearest_neighbors/{id_client}")
async def get_nearest_neighbors(id_client: int = Path(..., title="Client ID"), n_neighbors: int = 10):
    # Extraction des features du client en question
    client_data = data[data['SK_ID_CURR'] == id_client][relevant_features]

    # Initialisation du modèle des plus proches voisins
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)

    # Entraînement du modèle sur l'ensemble des données
    nn_model.fit(data[relevant_features])

    # Recherche des plus proches voisins du client
    _, indices = nn_model.kneighbors(client_data)

    # Récupération du dataframe des plus proches voisins
    nearest_neighbors_data = data.iloc[indices[0]]

    # S'assurer que la colonne 'SK_ID_CURR' est incluse dans la réponse
    nearest_neighbors_data = nearest_neighbors_data.reset_index(drop=True)

    return nearest_neighbors_data.to_dict(orient="records")

# Définition de la route pour récupérer les valeurs SHAP par client
@app.get("/shap_values/{id_client}")
async def get_shap_values_by_client(id_client: int = Path(..., title="Client ID")):
    # Filtre les données du client en question
    client_data = data[data['SK_ID_CURR'] == id_client]
    client_data = client_data[relevant_features]

    # Calcul des valeurs SHAP pour le client
    shap_values = explainer.shap_values(client_data)

    # Conversion des valeurs SHAP en un objet JSON
    shap_values_json = {
        "features": relevant_features,
        "values": shap_values[0].tolist()
    }

    return shap_values_json

# Définition d'une route pour obtenir les valeurs Shap de l'ensemble des données
@app.get("/shap")
async def get_shap_values():
    # Définir la taille du sous-échantillon
    subsample_size = 500

    # Sélectionner un sous-échantillon aléatoire du jeu de données
    subsample = data.sample(n=subsample_size, random_state=42)

    # Calculer les valeurs SHAP pour le sous-échantillon
    shap_values_all = explainer.shap_values(subsample[relevant_features])

    # Conversion des valeurs Shap en un objet JSON
    shap_values_json_all = {
        "features": relevant_features,
        "values": shap_values_all[0].tolist()
    }

    return shap_values_json_all

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
# Chemin vers le fichier HTML de dérive des données
data_drift_file_path = Path("./data_drift.html")

# Définissez le dossier statique pour les fichiers HTML
app.mount("/static", StaticFiles(directory=str(data_drift_file_path.parent), html=True), name="static")

# Route pour récupérer le fichier de dérive des données
@app.get("/data_drift_html")
async def get_data_drift_html():
    return FileResponse(data_drift_file_path, media_type="text/html")
