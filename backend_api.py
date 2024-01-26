# Importation des bibliothèques
#from flask import Flask
from fastapi import FastAPI, Path
import joblib
import pandas as pd
import shap
# from tensorflow.keras.models import load_model
import numpy as np
import io
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
shap.initjs()

# Initialisation d'une instance de l'API
app = FastAPI()

# Chargement du modèle de prédiction de crédit
clf = joblib.load("pipeline_depense_pred.joblib")
#model = joblib.load("mlflow_model/model.pkl")
# model = clf.named_steps["LogisticRegression"]

#Chargement des données
df = pd.read_csv('data_pred.csv', encoding ='utf-8')
target = df['TARGET']


# Définition des caractéristiques pertinentes (isolement des features non utilisées)
ignore_features = ['Unnamed: 0', 'SK_ID_CURR','INDEX', 'TARGET']
relevant_features = [col for col in df.columns if col not in ignore_features]

# # Création de l'explainer SHAP
#explainer = shap.LinearExplainer(clf, df)  # Utilisation de TreeExplainer au lieu de LinearExplainer
#masker = shap.utils.sample(df[relevant_features])
# background_data = shap.utils.sample(df[relevant_features])
# explainer = shap.Explainer(clf, background_data)  # Utilisation de TreeExplainer au lieu de LinearExplainer

@app.get('/')
def get_data():
    return {'welcome': 'Hello API'}

#####
# Définition de la route pour obtenir la liste des identifiants clients
@app.get("/client_ids")
def get_client_ids():
    clients_ids =  df['SK_ID_CURR'].tolist()
    return clients_ids

#####
# Définition de la route de prédiction de crédit ("/credit/{id_client}")
@app.get("/credit/{id_client}")
def predict_credit(id_client: int = Path(..., title="Client ID")):
    # Filtre sur les données du client en question
    X = df[df['SK_ID_CURR'] == id_client]
    X = X[relevant_features]

    # Calcul de la prédiction et de la probabilité de prédiction
    proba = clf.predict_proba(X)
    prediction = clf.predict(X)

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
        'Conclusion': interpretation,
    }
    # Retour de la réponse de la prédiction
    return pred_proba


#####
# Définition de la route pour calculer les plus proches voisins du client_id
@app.get("/nearest_neighbors/{id_client}")
async def get_nearest_neighbors(id_client: int = Path(..., title="Client ID"), n_neighbors: int = 10):
    # Extraction des features du client en question
    client_data = df[df['SK_ID_CURR'] == id_client][relevant_features]

    # Initialisation du modèle des plus proches voisins
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)

    # Entraînement du modèle sur l'ensemble des données
    nn_model.fit(df[relevant_features])

    # Recherche des plus proches voisins du client
    _, indices = nn_model.kneighbors(client_data)

    # Récupération du dataframe des plus proches voisins
    nearest_neighbors_df = df.iloc[indices[0]]

    list_id_nearest = nearest_neighbors_df["SK_ID_CURR"]
    # S'assurer que la colonne 'SK_ID_CURR' est incluse dans la réponse
    nearest_neighbors_df = nearest_neighbors_df.reset_index(drop=True)

    return nearest_neighbors_df.to_dict(orient="records")

####
# #Data drift_plot
# # Route pour récupérer le fichier de dérive des données
# @app.get("/data_drift_html")
# async def get_data_drift_html():
#     #return FileResponse(data_drift_file_path, media_type="text/html")
#     # Read file and keep in variable
#     with open("data_drift.html",'r') as f:
#         html_data = f.read()
#     ## Show in webpage
#     st.header("Show an external HTML")
#     st.components.v1.html(html_data,height=200)

####
# Définition de la route pour récupérer les valeurs SHAP par client
# @app.get("/shap_values/{id_client}")
# async def get_shap_values_by_client(id_client: int = Path(..., title="Client ID")):
#     # Filtre les données du client en question
#     client_data = df[df['SK_ID_CURR'] == id_client]
#     client_data = client_data[relevant_features]
#
#     # Calcul des valeurs SHAP pour le client
#     shap_values = explainer.shap_values(client_data)
#
#     # Conversion des valeurs SHAP en un objet JSON
#     shap_values_json = {
#         "features": relevant_features,
#         "values": shap_values[0].tolist()
#     }
#
#     return shap_values_json


#if __name__ == '__main__':
#    app.run(debug=True)
