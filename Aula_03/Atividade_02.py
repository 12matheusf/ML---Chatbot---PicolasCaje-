import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# 1. Preparação (Usando as features da Aula 02)

import pandas as pd

# --- ADICIONE ESTE BLOCO PARA CRIAR OS DADOS ---
data = {
    'qtd_exclamacao': [0, 5, 1, 10, 2, 8, 0, 12],
    'tem_palavra_alerta': [0, 1, 0, 1, 0, 1, 0, 1],
    'tamanho_msg': [10, 50, 15, 100, 20, 80, 5, 120],
    'rotulo': [0, 1, 0, 1, 0, 1, 0, 1] # 1: Alta Prioridade, 0: Comum
}
df = pd.DataFrame(data)
# ----------------------------------------------

# Agora o seu código original vai funcionar:
X = df[['qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg']]
# ... resto do código# Supondo que o df já tenha as colunas: 'qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg'
X = df[['qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg']]
y = df['rotulo'] # 1: Alta Prioridade, 0: Comum

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Treinamento
clf = DecisionTreeClassifier(max_depth=3) # Limitamos a profundidade para ser legível
clf.fit(X_train, y_train)

# 3. Visualização da Lógica (Explicação para os alunos)
# Diferente de uma "caixa preta", a Árvore nos mostra as regras que criou
regras = export_text(clf, feature_names=['Exclamações', 'Palavra Alerta', 'Tamanho'])
print("--- Regras Decididas pelo Modelo ---")
print(regras)
