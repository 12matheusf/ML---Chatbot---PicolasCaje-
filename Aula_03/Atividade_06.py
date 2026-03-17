import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# DATASET
data = {
    'tempo_conversa_min': [5, 12, 2, 20, 15, 8, 30, 10, 5, 25],
    'mensagens_trocadas': [10, 25, 4, 45, 30, 12, 60, 18, 9, 50],
    'custo_ads_real': [15.5, 32.0, 8.0, 55.0, 40.5, 20.0, 80.0, 25.0, 14.0, 65.0]
}
df = pd.DataFrame(data)

# --- DESAFIO DE REGRESSÃO MÚLTIPLA ---
# O objetivo é prever o 'custo_ads_real' usando 'tempo_conversa_min' E 'mensagens_trocadas'.

# --- ESPAÇO PARA O ALUNO DESENVOLVER ---

# 1. Defina X e y
# X contém as colunas preditoras (tempo e mensagens)
X = df[['tempo_conversa_min', 'mensagens_trocadas']]
# y é o que queremos prever (custo)
y = df['custo_ads_real']

# 2. Crie o modelo LinearRegression()
modelo = LinearRegression()

# 3. Treine o modelo
modelo.fit(X, y)

# 4. Use .predict([[15, 35]]) para prever o custo de uma conversa de 15min com 35 mensagens.
entrada = pd.DataFrame([[15, 35]], columns=['tempo_conversa_min', 'mensagens_trocadas'])
predicao = modelo.predict(entrada)

print(f"Previsão de custo para 15min e 35 mensagens: R$ {predicao[0]:.2f}")