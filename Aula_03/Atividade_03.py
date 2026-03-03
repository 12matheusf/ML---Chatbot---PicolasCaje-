import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Criando os dados base (necessário para o df existir)
data = {
    'tamanho_msg': [10, 50, 100, 150, 200, 250, 300, 400],
}
df = pd.DataFrame(data)

# 1. Criando o alvo (Target) com um pouco de ruído aleatório para ser realista
df['tempo_real_espera'] = (df['tamanho_msg'] * 0.5) + 5 + np.random.normal(0, 2, len(df))

X = df[['tamanho_msg']] 
y = df['tempo_real_espera']

# 2. Treinamento
reg = LinearRegression()
reg.fit(X, y)

# 3. Resultados
print("--- Resultado da Regressão Linear ---")
print(f"Intercepto (Tempo Base): {reg.intercept_:.2f} min")
print(f"Coeficiente (Aumento por Caractere): {reg.coef_[0]:.2f} min")

# Teste
previsao = reg.predict([[200]])
print(f"\nPrevisão para 200 caracteres: {previsao[0]:.2f} minutos")