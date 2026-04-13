"""
╔══════════════════════════════════════════════════════════════════╗
║      BLOCO 2 — PIPELINE COMPLETO DE MACHINE LEARNING            ║
║      Sistema de Classificação de Risco Clínico                  ║
║      Projeto Acadêmico — Machine Learning Aplicado à Saúde      ║
╚══════════════════════════════════════════════════════════════════╝

ETAPAS DO PIPELINE:
    1.  Leitura e exploração do dataset
    2.  Separação de features (X) e target (y)
    3.  Divisão treino / teste (80% / 20%)
    4.  Normalização com StandardScaler
    5.  Treinamento: Regressão Logística, Random Forest, KNN
    6.  Avaliação: Acurácia, Precision, Recall, F1-Score
    7.  Validação cruzada k-fold (k=5, StratifiedKFold)
    8.  Tabela comparativa e seleção do melhor modelo
    9.  Visualizações: barras, matriz de confusão, curva ROC
    10. Predição de novo paciente com probabilidades

PRÉ-REQUISITO:
    Execute antes: python 1_gerar_dataset.py
    Isso criará o arquivo pacientes.csv necessário.

BIBLIOTECAS:
    pandas, numpy, scikit-learn, matplotlib
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTAÇÕES — organizadas por categoria
# ──────────────────────────────────────────────────────────────────

# Dados e cálculo
import numpy as np
import pandas as pd

# Visualização
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Silencia avisos de convergência (ex: Regressão Logística)
import warnings
warnings.filterwarnings("ignore")

# ── Pré-processamento ──
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler, label_binarize

# ── Modelos ──
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.neighbors     import KNeighborsClassifier

# ── Métricas de avaliação ──
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 1 — LEITURA E EXPLORAÇÃO DO DATASET                     ║
# ╚══════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("  ETAPA 1 — LEITURA E EXPLORAÇÃO DO DATASET")
print("=" * 65)

# Lê o CSV gerado pelo script anterior
df = pd.read_csv("pacientes.csv")

print(f"\n  Shape             : {df.shape}  ({df.shape[0]} linhas × {df.shape[1]} colunas)")
print(f"  Tipos de dados    :\n{df.dtypes.to_string()}")
print(f"\n  Valores nulos por coluna:\n{df.isnull().sum().to_string()}")

print(f"\n  Distribuição da variável alvo 'risco':")
rotulos = {0: "0 — Baixo", 1: "1 — Médio", 2: "2 — Alto"}
for classe, qtd in df["risco"].value_counts().sort_index().items():
    print(f"    {rotulos[classe]:<14}: {qtd} registros ({qtd/len(df)*100:.1f}%)")

print(f"\n  Estatísticas descritivas:")
print(df.drop(columns="nome").describe().round(2).to_string())


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 2 — SEPARAÇÃO FEATURES (X) E TARGET (y)                 ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 2 — SEPARAÇÃO FEATURES (X) E TARGET (y)")
print("=" * 65)

# Removemos "nome" (identificador, sem poder preditivo) e "risco" (target)
# As features são as 5 variáveis biomédicas
X = df.drop(columns=["nome", "risco"])
y = df["risco"]

print(f"\n  Features preditivas : {list(X.columns)}")
print(f"  Shape de X          : {X.shape}")
print(f"  Shape de y          : {y.shape}")
print(f"  Classes em y        : {sorted(y.unique())}  → 0=Baixo, 1=Médio, 2=Alto")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 3 — DIVISÃO TREINO / TESTE                               ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 3 — DIVISÃO TREINO / TESTE  (80% / 20%)")
print("=" * 65)

# test_size=0.20  → 20% para teste (400 amostras), 80% para treino (1600)
# stratify=y      → garante que a proporção das classes seja mantida
#                   em ambos os conjuntos (muito importante em classificação)
# random_state=42 → semente para reproducibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n  Amostras de treino  : {len(X_train)}  ({len(X_train)/len(df)*100:.0f}%)")
print(f"  Amostras de teste   : {len(X_test)}   ({len(X_test)/len(df)*100:.0f}%)")

print(f"\n  Distribuição no treino:")
for c, v in y_train.value_counts().sort_index().items():
    print(f"    Classe {c}: {v} amostras ({v/len(y_train)*100:.1f}%)")

print(f"\n  Distribuição no teste:")
for c, v in y_test.value_counts().sort_index().items():
    print(f"    Classe {c}: {v} amostras ({v/len(y_test)*100:.1f}%)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 4 — NORMALIZAÇÃO COM STANDARDSCALER                     ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 4 — NORMALIZAÇÃO (StandardScaler)")
print("=" * 65)

"""
POR QUE NORMALIZAR?
    Algoritmos como KNN e Regressão Logística são sensíveis à escala.
    Ex: colesterol varia de 130–330, IMC de 15–48. Sem normalização,
    colesterol domina o cálculo de distância no KNN.

COMO FUNCIONA o StandardScaler?
    Para cada feature: z = (x - média) / desvio_padrão
    Resultado: média ≈ 0 e desvio ≈ 1 em cada coluna

REGRA FUNDAMENTAL (evitar data leakage):
    fit_transform() apenas no TREINO — aprende média e desvio do treino.
    transform()     apenas no TESTE  — aplica os mesmos parâmetros.
    Nunca fazer fit no teste! Isso contaminaria o modelo com informação futura.
"""

scaler = StandardScaler()

# Aprende os parâmetros de normalização com os dados de treino e já transforma
X_train_scaled = scaler.fit_transform(X_train)

# Aplica exatamente os mesmos parâmetros (média/desvio do treino) ao teste
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Parâmetros aprendidos (do treino):")
for i, col in enumerate(X.columns):
    print(f"    {col:<12}: média = {scaler.mean_[i]:>8.3f}  |  desvio = {scaler.scale_[i]:>7.3f}")

print(f"\n  Verificação pós-escala (treino):")
print(f"    Média por coluna   : {X_train_scaled.mean(axis=0).round(4)}")
print(f"    Desvio por coluna  : {X_train_scaled.std(axis=0).round(4)}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 5 — TREINAMENTO DOS MODELOS                              ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 5 — TREINAMENTO DOS MODELOS")
print("=" * 65)

"""
MODELOS ESCOLHIDOS E JUSTIFICATIVA:

1. Regressão Logística
   - Modelo linear e interpretável
   - Bom para entender relação linear entre features e classes
   - Rápido e serve como baseline (referência mínima de desempenho)
   - max_iter=1000: necessário para convergência com dados normalizados

2. Random Forest
   - Ensemble de múltiplas árvores de decisão
   - Robusto a outliers e não precisa de normalização (usamos para comparação)
   - Geralmente o melhor em datasets clínicos tabulares
   - n_estimators=100: número de árvores no ensemble

3. KNN (K-Nearest Neighbors)
   - Classifica com base nos K vizinhos mais próximos
   - Totalmente dependente da normalização (usa distância euclidiana)
   - Simples, sem fase de treinamento real ("lazy learning")
   - k=5: 5 vizinhos (valor padrão equilibrado)
"""

modelos = {
    "Regressão Logística" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (K=5)"           : KNeighborsClassifier(n_neighbors=5)
}

# Treina cada modelo e guarda no dicionário modelos_treinados
modelos_treinados = {}
for nome_modelo, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)        # treina com dados normalizados
    modelos_treinados[nome_modelo] = modelo    # armazena modelo treinado
    print(f"  ✔  {nome_modelo} treinado com sucesso.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 6 — AVALIAÇÃO COM MÉTRICAS                               ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 6 — AVALIAÇÃO DAS MÉTRICAS (conjunto de TESTE)")
print("=" * 65)

"""
MÉTRICAS UTILIZADAS:

  Acurácia    → % de predições corretas (pode enganar em classes desbalanceadas)
  Precision   → dos que o modelo disse ser classe X, quantos realmente são?
  Recall      → dos que realmente são classe X, quantos o modelo encontrou?
  F1-Score    → média harmônica entre Precision e Recall (melhor métrica geral)
  
  average='weighted' → pondera cada classe pela sua proporção no dataset
                        ideal para datasets com classes desbalanceadas
"""

resultados = {}  # dicionário para armazenar métricas de cada modelo

for nome_modelo, modelo in modelos_treinados.items():
    # Gera predições no conjunto de teste
    y_pred = modelo.predict(X_test_scaled)

    # Calcula cada métrica
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Guarda resultados (incluindo y_pred para uso na matriz de confusão)
    resultados[nome_modelo] = {
        "Acurácia"  : acc,
        "Precision" : prec,
        "Recall"    : rec,
        "F1-Score"  : f1,
        "y_pred"    : y_pred
    }

    print(f"\n  ── {nome_modelo} ──────────────────────────")
    print(f"     Acurácia   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"     Precision  : {prec:.4f}")
    print(f"     Recall     : {rec:.4f}")
    print(f"     F1-Score   : {f1:.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 7 — VALIDAÇÃO CRUZADA K-FOLD (k=5)                      ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 7 — VALIDAÇÃO CRUZADA (StratifiedKFold, k=5)")
print("=" * 65)

"""
POR QUE USAR VALIDAÇÃO CRUZADA?
    Avaliar no conjunto de teste uma única vez pode ser "sortudo".
    Com k-fold, dividimos os dados em k partes (folds):
    - Em cada rodada, 1 fold é teste e os outros k-1 são treino
    - Repetimos k vezes, usando cada fold como teste uma vez
    - Calculamos a média e desvio dos k scores
    
    StratifiedKFold: mantém proporção das classes em cada fold
    
    Um modelo com alta média e baixo desvio é mais confiável.
"""

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome_modelo, modelo in modelos_treinados.items():
    # cross_val_score aplica o modelo nos k folds automaticamente
    scores = cross_val_score(
        modelo,
        X_train_scaled,   # usa apenas o treino para validação cruzada
        y_train,
        cv=kfold,
        scoring="accuracy"
    )

    resultados[nome_modelo]["CV_media"]  = scores.mean()
    resultados[nome_modelo]["CV_desvio"] = scores.std()

    print(f"\n  ── {nome_modelo} ──────────────────────────")
    print(f"     Scores por fold : {scores.round(4)}")
    print(f"     Média           : {scores.mean():.4f}  ± {scores.std():.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 8 — TABELA COMPARATIVA E SELEÇÃO DO MELHOR MODELO       ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 8 — COMPARAÇÃO DOS MODELOS")
print("=" * 65)

# Monta tabela comparativa com todas as métricas
colunas_exibir = ["Acurácia", "Precision", "Recall", "F1-Score", "CV_media", "CV_desvio"]
tabela = pd.DataFrame(
    {nome: {c: resultados[nome][c] for c in colunas_exibir} for nome in resultados}
).T.round(4)

tabela.columns = ["Acurácia", "Precision", "Recall", "F1-Score", "CV Média", "CV Desvio"]
print(f"\n{tabela.to_string()}")

# Melhor modelo: maior F1-Score no conjunto de teste (métrica mais robusta)
melhor_nome   = max(resultados, key=lambda k: resultados[k]["F1-Score"])
melhor_modelo = modelos_treinados[melhor_nome]

print(f"\n  🏆 Melhor modelo selecionado : {melhor_nome}")
print(f"     F1-Score no teste         : {resultados[melhor_nome]['F1-Score']:.4f}")
print(f"     CV Média (treino k-fold)  : {resultados[melhor_nome]['CV_media']:.4f} ± {resultados[melhor_nome]['CV_desvio']:.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 9 — VISUALIZAÇÕES                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 9 — GERANDO VISUALIZAÇÕES")
print("=" * 65)

# Paleta de cores consistente para todos os gráficos
CORES = ["#2E86AB", "#E84855", "#3BB273"]
nomes_modelos = list(resultados.keys())


# ── 9.1  GRÁFICO COMPARATIVO DE MÉTRICAS ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Sistema de Classificação de Risco Clínico\nComparação dos Modelos de Machine Learning",
    fontsize=13, fontweight="bold", y=1.02
)

# Barras com as 4 métricas lado a lado por modelo
metricas_plot = ["Acurácia", "Precision", "Recall", "F1-Score"]
x = np.arange(len(nomes_modelos))
largura = 0.2   # largura de cada barra

for i, metrica in enumerate(metricas_plot):
    valores = [resultados[n][metrica] for n in nomes_modelos]
    barras  = axes[0].bar(x + i * largura, valores, largura, label=metrica)

axes[0].set_title("Métricas por Modelo (Conjunto de Teste)", fontsize=11)
axes[0].set_xticks(x + largura * 1.5)
axes[0].set_xticklabels(nomes_modelos, rotation=10, ha="right", fontsize=9)
axes[0].set_ylabel("Score (0–1)")
axes[0].set_ylim(0, 1.15)
axes[0].legend(fontsize=8)
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
axes[0].grid(axis="y", alpha=0.3)

# Barras de Validação Cruzada com barra de erro (desvio padrão)
cv_medias  = [resultados[n]["CV_media"]  for n in nomes_modelos]
cv_desvios = [resultados[n]["CV_desvio"] for n in nomes_modelos]

bars = axes[1].bar(nomes_modelos, cv_medias, color=CORES, edgecolor="black",
                   linewidth=0.8, yerr=cv_desvios, capsize=6)
axes[1].set_title("Validação Cruzada — Acurácia Média ± Desvio (k=5)", fontsize=11)
axes[1].set_ylabel("Acurácia Média")
axes[1].set_ylim(0, 1.1)
axes[1].set_xticklabels(nomes_modelos, rotation=10, ha="right", fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

# Exibe valor acima de cada barra
for bar, val, dev in zip(bars, cv_medias, cv_desvios):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + dev + 0.02,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("comparacao_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Gráfico comparativo salvo → comparacao_modelos.png")


# ── 9.2  MATRIZ DE CONFUSÃO (melhor modelo) ──────────────────────
y_pred_melhor = resultados[melhor_nome]["y_pred"]
cm = confusion_matrix(y_test, y_pred_melhor)
labels_conf = ["Baixo (0)", "Médio (1)", "Alto (2)"]

"""
INTERPRETANDO A MATRIZ DE CONFUSÃO:
    - Linhas  = classe real
    - Colunas = classe predita pelo modelo
    - Diagonal principal = acertos
    - Fora da diagonal   = erros (tipo e quantidade)
    
    Exemplo: cm[1][2] = pacientes de risco MÉDIO classificados como ALTO
"""

fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_conf)
disp.plot(ax=ax_cm, colorbar=True, cmap="Blues", values_format="d")
ax_cm.set_title(
    f"Matriz de Confusão — {melhor_nome}\n"
    f"(Conjunto de Teste — {len(y_test)} amostras)",
    fontsize=11, fontweight="bold"
)
ax_cm.set_xlabel("Classe Predita", fontsize=10)
ax_cm.set_ylabel("Classe Real", fontsize=10)

plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Matriz de confusão salva → matriz_confusao.png")


# ── 9.3  CURVA ROC (One-vs-Rest, se modelo suportar) ─────────────
if hasattr(melhor_modelo, "predict_proba"):
    """
    CURVA ROC (Receiver Operating Characteristic):
        - Eixo X: Taxa de Falsos Positivos (FPR)
        - Eixo Y: Taxa de Verdadeiros Positivos (TPR / Recall)
        - AUC (Area Under Curve): quanto maior, melhor
          AUC = 0.5 → modelo aleatório (linha diagonal tracejada)
          AUC = 1.0 → modelo perfeito
        
        Para multiclasse: usamos One-vs-Rest (OvR)
        Cada classe é tratada como "positiva" e as outras como "negativas"
    """
    classes = [0, 1, 2]

    # label_binarize converte y em matriz binária para OvR
    # Ex: classe 1 → [0, 1, 0]
    y_test_bin = label_binarize(y_test, classes=classes)

    # predict_proba retorna probabilidades para cada classe
    y_score = melhor_modelo.predict_proba(X_test_scaled)

    cores_roc  = ["#2E86AB", "#E84855", "#3BB273"]
    labels_roc = ["Baixo risco (0)", "Risco médio (1)", "Risco alto (2)"]

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for i, (cor, lbl) in enumerate(zip(cores_roc, labels_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=cor, lw=2.5,
                    label=f"{lbl}  (AUC = {roc_auc:.3f})")

    # Linha diagonal = classificador aleatório (baseline)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, label="Aleatório (AUC = 0.5)")

    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11)
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11)
    ax_roc.set_title(
        f"Curva ROC — {melhor_nome}\n(Estratégia One-vs-Rest)",
        fontsize=12, fontweight="bold"
    )
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("curva_roc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✔  Curva ROC salva → curva_roc.png")
else:
    print(f"  ⚠  {melhor_nome} não suporta predict_proba. Curva ROC não gerada.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ETAPA 10 — PREDIÇÃO DE NOVO PACIENTE                          ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  ETAPA 10 — PREDIÇÃO DE NOVO PACIENTE FICTÍCIO")
print("=" * 65)

"""
FLUXO DA PREDIÇÃO (alinhado com o documento de arquitetura):
    Frontend → envia dados do paciente
    Backend  → coleta e formata os dados
    Modelo   → recebe features normalizadas, retorna classificação
    Saída    → classe (Baixo/Médio/Alto) + probabilidades por classe

IMPORTANTE: o novo paciente deve passar pela mesma normalização
            que foi aplicada durante o treinamento (mesmo scaler).
"""

# ── Dados do novo paciente (mesma ordem das features de X) ───────
# Features: ["idade", "glicose", "pressao", "imc", "colesterol"]

novo_paciente_raw = {
    "nome"       : "Helena",
    "idade"      : 67,
    "glicose"    : 152.0,   # acima de 126 → risco glicêmico
    "pressao"    : 147.0,   # acima de 140 → hipertensão
    "imc"        : 31.8,    # acima de 30  → obesidade
    "colesterol" : 258.0    # acima de 240 → dislipidemia
}

# Monta DataFrame com as features na mesma ordem do treino
X_novo = pd.DataFrame([{
    "idade"      : novo_paciente_raw["idade"],
    "glicose"    : novo_paciente_raw["glicose"],
    "pressao"    : novo_paciente_raw["pressao"],
    "imc"        : novo_paciente_raw["imc"],
    "colesterol" : novo_paciente_raw["colesterol"]
}])

# Exibe os dados brutos do paciente
print(f"\n  👤 Paciente: {novo_paciente_raw['nome']}")
print(f"  {'─' * 40}")
for col in X_novo.columns:
    print(f"    {col:<12}: {X_novo[col].values[0]}")

# Aplica a normalização (mesmos parâmetros do treino!)
X_novo_scaled = scaler.transform(X_novo)

# Predição da classe
classe_predita = melhor_modelo.predict(X_novo_scaled)[0]

# Predição das probabilidades (se o modelo suportar)
if hasattr(melhor_modelo, "predict_proba"):
    probabilidades = melhor_modelo.predict_proba(X_novo_scaled)[0]
else:
    # Fallback: decision_function normalizado para modelos sem predict_proba
    probabilidades = None

# ── Dicionário de resultados e mapeamentos ────────────────────────
mapa_classe = {0: "BAIXO RISCO", 1: "RISCO MÉDIO", 2: "RISCO ALTO"}
mapa_cor    = {0: "🟢", 1: "🟡", 2: "🔴"}
mapa_acao   = {
    0: "Acompanhamento de rotina anual.",
    1: "Consulta em até 3 meses. Mudanças de hábito recomendadas.",
    2: "Encaminhamento urgente para especialista. Monitoramento intensivo."
}

# ── Exibição do resultado ─────────────────────────────────────────
print(f"\n  {'═' * 50}")
print(f"  RESULTADO DA CLASSIFICAÇÃO DE RISCO")
print(f"  {'═' * 50}")
print(f"\n  Paciente          : {novo_paciente_raw['nome']}")
print(f"  Modelo utilizado  : {melhor_nome}")
print(f"\n  {mapa_cor[classe_predita]}  Classificação     : {mapa_classe[classe_predita]}")

if probabilidades is not None:
    print(f"\n  Probabilidades por classe:")
    for i, (classe, prob) in enumerate(zip(["Baixo", "Médio", "Alto"], probabilidades)):
        barra = "█" * int(prob * 30)
        print(f"    {classe:<6} (classe {i}): {prob*100:>6.2f}%  {barra}")

print(f"\n  📋 Conduta sugerida:")
print(f"     {mapa_acao[classe_predita]}")
print(f"\n  {'═' * 50}")

# ── Análise dos fatores de risco do paciente ──────────────────────
print(f"\n  🔍 Análise dos fatores de risco presentes:")
fatores = []
if novo_paciente_raw["glicose"]    >= 126: fatores.append("  ⚠  Glicose elevada (≥ 126 mg/dL)")
if novo_paciente_raw["pressao"]    >= 140: fatores.append("  ⚠  Pressão arterial elevada (≥ 140 mmHg)")
if novo_paciente_raw["imc"]        >= 30:  fatores.append("  ⚠  IMC indicando obesidade (≥ 30)")
if novo_paciente_raw["colesterol"] >= 240: fatores.append("  ⚠  Colesterol alto (≥ 240 mg/dL)")
if novo_paciente_raw["idade"]      >= 60:  fatores.append("  ⚠  Idade acima de 60 anos")

if fatores:
    for fator in fatores:
        print(fator)
    print(f"\n  Total de fatores de risco : {len(fatores)} de 5 possíveis")
else:
    print("  ✔  Nenhum fator de risco identificado.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  RESUMO FINAL DO PIPELINE                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  RESUMO FINAL DO PIPELINE")
print("=" * 65)

print(f"""
  Dataset               : 2000 registros × 5 features biomédicas
  Divisão               : 1600 treino / 400 teste (80/20)
  Normalização          : StandardScaler (fit apenas no treino)
  Modelos treinados     : Regressão Logística, Random Forest, KNN
  Validação             : StratifiedKFold (k=5) no conjunto de treino
  Melhor modelo         : {melhor_nome}
    → F1-Score (teste)  : {resultados[melhor_nome]['F1-Score']:.4f}
    → CV Média          : {resultados[melhor_nome]['CV_media']:.4f} ± {resultados[melhor_nome]['CV_desvio']:.4f}

  Arquivos gerados:
    📊 comparacao_modelos.png
    📊 matriz_confusao.png
    📊 curva_roc.png

  Pipeline concluído com sucesso! ✅
""")
print("=" * 65)
