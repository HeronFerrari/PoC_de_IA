import os
import numpy as np
import pandas as pd

# Descobre a pasta onde o app.py está e cria o caminho correto para a pasta data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_DADOS = os.path.join(BASE_DIR, "data", "dados_bezerros.csv")

# Garante que a pasta data existe
os.makedirs("data", exist_ok=True)

# Configura a semente aleatória para resultados reproduzíveis
np.random.seed(42)
n_amostras = 200

# Simulação dos dados dos bezerros
id_brincos = [f"BRINCO_{i:03d}" for i in range(1, n_amostras + 1)]
dias = np.random.randint(10, 150, size=n_amostras)

# Peso: base de 35kg + ganho médio de 0.75kg por dia + ruído
peso_ruido = np.random.normal(0, 5, size=n_amostras)
peso_kg = 35 + (0.75 * dias) + peso_ruido

# Custo: base de R$150 + custo médio de R$4.20 por dia + ruído
custo_ruido = np.random.normal(0, 25, size=n_amostras)
custo_acumulado = 150 + (4.2 * dias) + custo_ruido

# Montando o DataFrame
df = pd.DataFrame(
    {
        "ID_Animal": id_brincos,
        "Dias_De_Vida": dias,
        "Peso_Atual_KG": np.round(peso_kg, 2),
        "Custo_Acumulado_R$": np.round(custo_acumulado, 2),
    }
)

# Salva populando o arquivo correto
df.to_csv("data/dados_bezerros.csv", index=False)
print("✅ Arquivo 'data/dados_bezerros.csv' gerado com sucesso!")