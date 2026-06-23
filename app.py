import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Configuração da página do Streamlit
st.set_page_config(page_title="SmartCalf Predictor", page_icon="🐮", layout="wide")

st.title("SmartCalf Predictor 🐮")
st.markdown("---")

# 2. Carregamento dos dados simulados
try:
    df = pd.read_csv("data/dados_bezerros.csv")
    if df.empty:
        raise pd.errors.EmptyDataError

    # Layout em colunas para organizar a tela
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Amostra da Base de Dados Simulada")
        st.dataframe(df.head(10), use_container_width=True)
        
    # 3. Treinamento Automático da IA em Background para a PoC
    # Modelo 1: Predição de Peso com base nos Dias de Vida
    X = df[['Dias_De_Vida']]
    y_peso = df['Peso_Atual_KG']
    modelo_peso = LinearRegression().fit(X, y_peso)
    
    # Modelo 2: Predição de Custos com base nos Dias de Vida
    y_custo = df['Custo_Acumulado_R$']
    modelo_custo = LinearRegression().fit(X, y_custo)
    
    with col2:
        st.subheader("🔮 Simulador de Projeções Futuras (IA)")
        st.write("Insira a idade planejada para calcular o ganho de peso e custos estimados:")
        
        # Entrada interativa do usuário
        dias_futuros = st.slider("Dias de Vida Planejados:", min_value=1, max_value=365, value=90)
        
        # Realizando as predições usando a Regressão Linear
        entrada_modelo = np.array([[dias_futuros]])
        peso_predito = modelo_peso.predict(entrada_modelo)[0]
        custo_predito = modelo_custo.predict(entrada_modelo)[0]
        
        # Exibindo os resultados em blocos de métricas visuais
        m1, m2 = st.columns(2)
        m1.metric(label="Peso Estimado (KG)", value=f"{peso_predito:.2f} kg")
        m2.metric(label="Custo Acumulado Estimado (R$)", value=f"R$ {custo_predito:.2f}")

    st.markdown("---")
    st.subheader("📈 Comportamento de Evolução Temporal")
    
    # 4. Gráfico da PoC demonstrando as linhas de tendência encontradas pela IA
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico Peso
    ax[0].scatter(df['Dias_De_Vida'], df['Peso_Atual_KG'], color='gray', alpha=0.5, label='Dados Simulados')
    linha_X = np.linspace(10, 365, 100).reshape(-1, 1)
    ax[0].plot(linha_X, modelo_peso.predict(linha_X), color='#1E88E5', linewidth=3, label='Predição da IA')
    ax[0].set_title('Curva de Projeção de Peso')
    ax[0].set_xlabel('Dias de Vida')
    ax[0].set_ylabel('Peso (KG)')
    ax[0].grid(True)
    ax[0].legend()
    
    # Gráfico Custos
    ax[1].scatter(df['Dias_De_Vida'], df['Custo_Acumulado_R$'], color='gray', alpha=0.5, label='Dados Simulados')
    ax[1].plot(linha_X, modelo_custo.predict(linha_X), color='#D81B60', linewidth=3, label='Predição da IA')
    ax[1].set_title('Curva de Evolução de Custos')
    ax[1].set_xlabel('Dias de Vida')
    ax[1].set_ylabel('Custo Acumulado (R$)')
    ax[1].grid(True)
    ax[1].legend()
    
    st.pyplot(fig)

except (FileNotFoundError, pd.errors.EmptyDataError):
    st.warning(
        "⚠️ A base de dados 'dados_bezerros.csv' não foi encontrada ou está vazia!"
    )
    st.info("Execute o comando no terminal para gerar os dados: `python gerar_dados.py`")