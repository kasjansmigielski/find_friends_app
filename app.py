import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import json


MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

#używamy w funkcji aby skorzystać z dekoratora st.cache_resource
#to sprawi, że wczytane dane zostaną TYLKO RAZ

@st.cache_resource
def get_all_paricipants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep= ';')
    df_with_clusters = predict_model(model, data= all_df)
    
    return df_with_clusters

@st.cache_resource
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, 'r', encoding= 'utf-8') as f:
        data = json.loads(f.read())

    return data


with st.sidebar:
    st.markdown('Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania - opisz siebie:')

    age = st.selectbox('Wiek', ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown' ])
    edu_level = st.selectbox('Wykształcenie', ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

person_df = pd.DataFrame([
    {
    'age' : age,
    'edu_level': edu_level,
    'fav_animals': fav_animals,
    'fav_place': fav_place,
    'gender': gender
}
])

#JEDNOKROTNE wczytanie danych

model = get_model()
all_df = get_all_paricipants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data= person_df)['Cluster'].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f'Najbliżej Ci do grupy: {predicted_cluster_data["name"]}')

#wygenerowane tematyczne obrazki od openart ai
if predicted_cluster_id == 'Cluster 0':
    st.image('https://cdn.openart.ai/uploads/image_Vk27z142_1727548483511_raw.jpg')
elif predicted_cluster_id == 'Cluster 1':
    st.image('https://cdn.openart.ai/uploads/image_ABB9jZ7N_1727549147415_raw.jpg')
elif predicted_cluster_id == 'Cluster 2':
    st.image('https://cdn.openart.ai/uploads/image_SqHUbKrC_1727547617784_raw.jpg')
elif predicted_cluster_id == 'Cluster 3':
    st.image('https://cdn.openart.ai/uploads/image__4hewy11_1727549274916_raw.jpg')
elif predicted_cluster_id == 'Cluster 4':
    st.image('https://cdn.openart.ai/uploads/image_vxalAIOP_1727548110511_raw.jpg')
elif predicted_cluster_id == 'Cluster 5':
    st.image('https://cdn.openart.ai/uploads/image_XMDAv_bp_1727548278005_raw.jpg')
elif predicted_cluster_id == 'Cluster 6':
    st.image('https://cdn.openart.ai/uploads/image_qQJBBAh4_1727548377342_raw.jpg')
else:
    st.image('https://cdn.openart.ai/uploads/image_rltFsKa8_1727548484168_raw.jpg')

st.markdown(predicted_cluster_data['description'])
st.write('---')
same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
st.metric('Liczba Twoich znajomych:', len(same_cluster_df))

#rysowanie histogramów dla osób z tego samego klastra co ja

st.header('Rozkład danych dla osób z tej samej grupy:')
fig = px.histogram(same_cluster_df.sort_values('age'), x='age')
fig.update_layout(
    title = 'Rozkład wieku w grupie',
    xaxis_title = 'Wiek',
    yaxis_title = 'Liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='edu_level')
fig.update_layout(
    title = 'Rozkład wykształcenia w grupie',
    xaxis_title = 'Wykształcenie',
    yaxis_title = 'Liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='fav_animals')
fig.update_layout(
    title = 'Rozkład ulubionych zwierząt w grupie',
    xaxis_title = 'Ulubione zwierzęta',
    yaxis_title = 'Liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='fav_place')
fig.update_layout(
    title = 'Rozkład ulubionych miejsc w grupie',
    xaxis_title = 'Ulubione miejsce',
    yaxis_title = 'Liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='gender')
fig.update_layout(
    title = 'Rozkład płci w grupie',
    xaxis_title = 'Płeć',
    yaxis_title = 'Liczba osób'
)
st.plotly_chart(fig)


