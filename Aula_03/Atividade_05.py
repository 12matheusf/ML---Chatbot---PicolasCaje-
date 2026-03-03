import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# DICA 1: Carregar o arquivo gerado na Parte A
df = pd.read_csv('dados_intencoes.csv')

# DICA 2: Separar as features (X) do alvo (y)
X = df.drop('label', axis=1)
y = df['label']

# --- INÍCIO DO TRABALHO DO ALUNO ---

# PASSO 1: Divida os dados em treino e teste (use test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASSO 2: Instancie o DecisionTreeClassifier e treine com .fit()
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# PASSO 3: Gere predições e use classification_report para ver o resultado
predicoes = modelo.predict(X_test)

print("Relatório de Classificação:\n")
print(classification_report(y_test, predicoes))