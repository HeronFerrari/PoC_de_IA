import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(page_title="SmartCalf Predictor", page_icon="🐮", layout="wide")

st.title("SmartCalf Predictor 🐮")
st.markdown("---")

st.subheader("Visualização dos Dados do Rebanho")

# Exemplo de leitura de dados
try:
    df = pd.read_csv("data/dados_bezerros.csv")
    st.dataframe(df.head(10), use_container_width=True)
except FileNotFoundError:
    st.warning("Base de dados 'dados_bezerros.csv' não encontrada na pasta data/. Crie os dados primeiro!")