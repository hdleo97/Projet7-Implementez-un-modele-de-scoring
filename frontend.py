# DashBoard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import joblib
import plotly.express as px
import imblearn
#from tensorflow.keras.models import load_model
#Librairie de l'API
import requests
import json
from sklearn.linear_model import LogisticRegression
from zipfile import ZipFile

def main():
    api_main_url = 'http://localhost:8000'
    # response = requests.get(api_main_url + '/')
    # welcome = response.json()
    # st.write(welcome['welcome'])

    z = ZipFile("application_train.zip")

    #Chargement du model
    clf = joblib.load("pipeline_depense_pred.joblib")
    model = LogisticRegression()

    data = pd.read_csv(z.open('application_train.csv')) #pd.read_csv('application_train.csv', encoding ='utf-8')
    df = pd.read_csv('data_pred.csv', encoding ='utf-8')
    target = data['TARGET']
    # data_pred = data.join(data_test)
    # data_pred = data_pred.drop('TARGET')


    # Définition des caractéristiques pertinentes (isolement des features non utilisées)
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR','INDEX', 'TARGET']
    relevant_features = [col for col in data.columns if col not in ignore_features]

    ########Fonction######

    def load_model():
        model = joblib.load("pipeline_depense_pred.joblib") #mlflow_model/model.pkl
        return model

    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]
        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets

    def identite_client(data, id):
        data_client = data[data['SK_ID_CURR'] == int(id)]
        return data_client

    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    def load_prediction(id):
        api_url = api_main_url + f"/credit/{id}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            return []

    def get_client_ids():
        api_url = api_main_url + "/client_ids"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            return []

    # Fonction pour récupérer les valeurs SHAP pour un client spécifique
    def get_shap_values_by_client(client_id):
        api_url = api_main_url + "/shap_values/"
        response = requests.get(f"{api_url}{client_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching SHAP values. Status Code: {response.status_code}")
            return None

    # Fonction pour afficher le graphique de feature importance
    def plot_feature_importance(features, values):
        fig = px.bar(x=features, y=values, labels={'x': 'Feature', 'y': 'SHAP Value'},
                     title='Feature Importance - SHAP Values')
        st.plotly_chart(fig)

    # Définition de la route pour récupérer les valeurs SHAP par client
    def get_shap_values_by_client(client_id):
        # Filtre les données du client en question
        client_data = data[data['SK_ID_CURR'] == client_id]
        client_data = client_data[relevant_features]

        # Créer l'explainer avec le masker
        # masker = shap.maskers.Independent(data=clf.named_steps['preprocessor'].transform(X), max_samples=100)
        # explainer = shap.LinearExplainer(clf.named_steps['classifier'], masker)

        # Extraire le modèle réel du pipeline
        #model = clf.named_steps['model']  # Remplacez 'classifier' par la clé appropriée
        explainer = shap.LinearExplainer(model, client_data)
        #explainer = shap.LinearExplainer(clf)  # Utilisation de TreeExplainer au lieu de LinearExplainer

        # Calcul des valeurs SHAP pour le client
        shap_values = explainer.shap_values(client_data)

        # Conversion des valeurs SHAP en un objet JSON
        shap_values_json = {
            "features": relevant_features,
            "values": shap_values[0].tolist()
        }
        return shap_values_json

    def get_nearest_neighbors(client_id):
        api_url = api_main_url + f"/nearest_neighbors/{client_id}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            return []

    def get_datadrift_plot():
        api_url = api_main_url + "/data_drift_html"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response
        else:
            return []

    id_client = get_client_ids()
    ####HEADER#####
    # st.image("Logo.png")

    # Créer un titre
    st.title("Tableau de bord de capacité de remboursement de crédit")

    #######################################
    # SIDEBAR
    #######################################

    #Title display
    st.sidebar.title("Dashboard Scoring Credit")

    #Customer ID selection
    st.sidebar.header("**Information général**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
    st.sidebar.markdown("<u>Nombre de crédits :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Average income
    st.sidebar.markdown("<u>Moyenne des revenu (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    #AMT CREDIT
    st.sidebar.markdown("<u>Moyen des sommes emprunté (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    #PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)


    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar
    st.write("Identifiant du client sélectionné:", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Affichage des informations client**")

    if st.checkbox("Afficher les informations ?"):

        infos_client = identite_client(data, chk_id)
        st.write("**Genre : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
        st.write("**Status familiale : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='Âge client', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)


        st.subheader("*Revenu (USD)*")
        st.write("**Revenu total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant de crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Rentes de crédit : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Montant de crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))

        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Revenu client', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)

        #Relationship Age / Income Total interactive plot
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/365).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor':'#f0f0f0'},
                          title={'text':"Relationship Age / Income Total", 'x':0.5, 'xanchor': 'center'},
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    #Customer solvability display
    st.header("**Customer file analysis**")
    prediction = load_prediction(chk_id)
    st.write("**Prédiction** : ", prediction['Prédiction'])
    st.write("Probabilité de réalisation: {:.0f} %".format(round(float(prediction['Probabilité'])*100, 2)))
    st.write("Score du client :", round((prediction['Probabilité'])*100,2))
    st.write("Interpretation: ",format(prediction['Conclusion']))

    #Compute decision according to the best threshold
    if prediction['Prédiction'] == 0 :
       decision = "<font color='green'>**PRÊT ACCEPTER**</font>"
    else:
       decision = "<font color='red'>**PRÊT REJETE**</font>"

    st.write("**Decision** *(with threshold xx%)* **: **", decision, unsafe_allow_html=True)

    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))


    #Feature importance / description
    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):

        st.header("Feature importance globale")
        st.image("feature_importance_globale.png")
        # Récupération des valeurs SHAP pour le client spécifié
        # shap_data = get_shap_values_by_client(chk_id)

        # Affichage du graphique de feature importance si les valeurs SHAP sont disponibles
        # if shap_data:
        #     features = shap_data.get("features", [])
        #     values = shap_data.get("values", [])
        #     st.header("Features importance locale")
        #     plot_feature_importance(features, values)
        #
        # else:
        #     st.markdown("<i>…</i>", unsafe_allow_html=True)


    #Similar customer files display
    chk_voisins = st.checkbox("Show similar customer files ?")

    if chk_voisins:
        knn_data = get_nearest_neighbors(chk_id)
        st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
        st.dataframe(knn_data)
        st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    st.markdown('***')
    st.markdown("Thanks ❤️")


if __name__ == '__main__':
    main()
