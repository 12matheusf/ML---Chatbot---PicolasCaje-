"""
╔══════════════════════════════════════════════════════════════════╗
║       BLOCO 1 — GERADOR DE DATASET SINTÉTICO DE PACIENTES       ║
║       Sistema de Classificação de Risco Clínico                 ║
║       Projeto Acadêmico — Machine Learning Aplicado à Saúde     ║
╚══════════════════════════════════════════════════════════════════╝

OBJETIVO:
    Gerar 2000 registros fictícios de pacientes com variáveis
    biomédicas realistas. A variável alvo "risco" é definida por
    regras clínicas baseadas em diretrizes médicas reais.

VARIÁVEIS GERADAS:
    nome        → Nome fictício simples (sem sobrenome)
    idade       → 18 a 99 anos
    glicose     → mg/dL  (referência: normal < 100, pré-diabético 100–125, diabético ≥ 126)
    pressao     → mmHg   (normal < 120, elevada 120–139, hipertensão ≥ 140)
    imc         → kg/m²  (abaixo do peso < 18.5, normal 18.5–24.9, sobrepeso 25–29.9, obeso ≥ 30)
    colesterol  → mg/dL  (desejável < 200, limítrofe 200–239, alto ≥ 240)

VARIÁVEL ALVO:
    risco  →  0 = Baixo risco | 1 = Risco médio | 2 = Risco alto

BIBLIOTECAS:
    numpy  — geração de números aleatórios com controle de seed
    pandas — estrutura de dados tabular e exportação CSV
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTAÇÕES
# ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO INICIAL
# seed=42 garante que os mesmos dados sejam gerados sempre que
# o script for executado — essencial para reproducibilidade
# ──────────────────────────────────────────────────────────────────
np.random.seed(42)

N = 2000  # número exato de registros a gerar


# ══════════════════════════════════════════════════════════════════
# PASSO 1 — NOMES FICTÍCIOS
# Lista de nomes brasileiros comuns, sem sobrenomes.
# np.random.choice sorteia aleatoriamente com reposição.
# ══════════════════════════════════════════════════════════════════
nomes_masculinos = [
    "Carlos", "Lucas", "Pedro", "Marcos", "João", "Rafael", "Bruno",
    "Diego", "Felipe", "André", "Thiago", "Gustavo", "Eduardo", "Ricardo",
    "Rodrigo", "Henrique", "Alexandre", "Fernando", "Leandro", "Fábio",
    "Márcio", "Roberto", "Sérgio", "Daniel", "Mateus", "Samuel", "Paulo",
    "Leonardo", "Vinicius", "Gabriel"
]

nomes_femininos = [
    "Ana", "Maria", "Juliana", "Fernanda", "Camila", "Patrícia", "Beatriz",
    "Larissa", "Vanessa", "Gabriela", "Renata", "Aline", "Tatiana", "Cláudia",
    "Mônica", "Priscila", "Sandra", "Helena", "Letícia", "Simone",
    "Mariana", "Amanda", "Cristina", "Débora", "Elaine", "Fabiana",
    "Giovana", "Isabela", "Karina", "Luana"
]

# Junta as duas listas e sorteia N nomes aleatoriamente
todos_nomes = nomes_masculinos + nomes_femininos
nomes = np.random.choice(todos_nomes, size=N, replace=True)


# ══════════════════════════════════════════════════════════════════
# PASSO 2 — GERAÇÃO DAS VARIÁVEIS BIOMÉDICAS
#
# As faixas foram definidas com base em referências clínicas:
#   - Sociedade Brasileira de Diabetes (SBD)
#   - VII Diretriz Brasileira de Hipertensão Arterial
#   - OMS / WHO para IMC
#   - Sociedade Brasileira de Cardiologia (colesterol)
#
# np.random.uniform(low, high) → distribuição uniforme contínua
# np.random.randint(low, high) → inteiro uniforme (high exclusive)
# np.round(array, decimais)    → arredonda para N casas decimais
# ══════════════════════════════════════════════════════════════════

# IDADE: inteiro entre 18 e 99 anos
# randint é exclusivo no limite superior → usamos 100
idade = np.random.randint(18, 100, size=N)

# GLICOSE: mg/dL — faixa de jejum de 70 (hipoglicemia leve) a 220
# Pacientes com 220 estão em estado de hiperglicemia significativa
glicose = np.round(np.random.uniform(70, 220, size=N), 1)

# PRESSÃO ARTERIAL SISTÓLICA: mmHg — de 80 (muito baixa) a 190
# Valores acima de 180 indicam crise hipertensiva
pressao = np.round(np.random.uniform(80, 190, size=N), 1)

# IMC: kg/m² — de 15.0 (desnutrição grave) a 48.0 (obesidade grau III)
imc = np.round(np.random.uniform(15.0, 48.0, size=N), 2)

# COLESTEROL TOTAL: mg/dL — de 130 (muito baixo) a 330 (alto risco CV)
colesterol = np.round(np.random.uniform(130, 330, size=N), 1)


# ══════════════════════════════════════════════════════════════════
# PASSO 3 — REGRA DE CLASSIFICAÇÃO DE RISCO
#
# Cada variável fora da faixa saudável contribui com 1 ponto.
# A soma total define a classe de risco do paciente.
#
# CRITÉRIOS (baseados em diretrizes clínicas):
#   Glicose ≥ 126        → + 1 pt  (limiar diagnóstico de DM2)
#   Pressão ≥ 140        → + 1 pt  (hipertensão estágio 1)
#   IMC ≥ 30             → + 1 pt  (obesidade grau I)
#   Colesterol ≥ 240     → + 1 pt  (hipercolesterolemia)
#   Idade ≥ 60           → + 1 pt  (maior risco cardiovascular)
#
# MAPEAMENTO:
#   0 pontos   → risco 0 (Baixo)
#   1–2 pontos → risco 1 (Médio)
#   3+ pontos  → risco 2 (Alto)
#
# .astype(int) converte True/False em 1/0 para somar os pontos
# ══════════════════════════════════════════════════════════════════

pontos = (
    (glicose    >= 126).astype(int) +  # risco glicêmico
    (pressao    >= 140).astype(int) +  # risco hipertensivo
    (imc        >= 30 ).astype(int) +  # risco obesidade
    (colesterol >= 240).astype(int) +  # risco dislipidemia
    (idade      >= 60 ).astype(int)    # risco etário
)

# Função que converte pontuação em categoria de risco
def pontuar_risco(p):
    """
    Recebe a pontuação de fatores de risco e retorna:
        0 → baixo risco
        1 → risco médio
        2 → risco alto
    """
    if p == 0:
        return 0
    elif p <= 2:
        return 1
    else:
        return 2

# Aplica a função para cada paciente usando list comprehension
risco = np.array([pontuar_risco(p) for p in pontos])


# ══════════════════════════════════════════════════════════════════
# PASSO 4 — MONTAGEM DO DATAFRAME PANDAS
#
# Cada chave do dicionário vira uma coluna no DataFrame.
# A ordem das colunas reflete a especificação funcional do sistema.
# ══════════════════════════════════════════════════════════════════
df = pd.DataFrame({
    "nome"       : nomes,
    "idade"      : idade,
    "glicose"    : glicose,
    "pressao"    : pressao,
    "imc"        : imc,
    "colesterol" : colesterol,
    "risco"      : risco
})


# ══════════════════════════════════════════════════════════════════
# PASSO 5 — INSPEÇÃO E VALIDAÇÃO DO DATASET
# ══════════════════════════════════════════════════════════════════
print("=" * 65)
print("  GERADOR DE DATASET — SISTEMA DE CLASSIFICAÇÃO DE RISCO")
print("=" * 65)

print(f"\n{'Total de registros gerados':.<40} {len(df)}")
print(f"{'Colunas':.<40} {list(df.columns)}")
print(f"\n{'─' * 65}")

print("\n📋 PRIMEIROS 10 REGISTROS:")
print(df.head(10).to_string(index=False))

print(f"\n{'─' * 65}")
print("\n📊 DISTRIBUIÇÃO DA VARIÁVEL ALVO (risco):")
contagem = df["risco"].value_counts().sort_index()
rotulos  = {0: "0 — Baixo risco", 1: "1 — Risco médio", 2: "2 — Risco alto"}
for classe, qtd in contagem.items():
    pct = qtd / N * 100
    print(f"   {rotulos[classe]:<22}: {qtd:>5} registros  ({pct:.1f}%)")

print(f"\n{'─' * 65}")
print("\n📈 ESTATÍSTICAS DESCRITIVAS DAS FEATURES NUMÉRICAS:")
print(df.drop(columns="nome").describe().round(2).to_string())

print(f"\n{'─' * 65}")
print("\n🔍 VERIFICAÇÃO DE QUALIDADE:")
print(f"   Valores nulos     : {df.isnull().sum().sum()}")
print(f"   Idades inválidas  : {((df['idade'] < 18) | (df['idade'] > 99)).sum()}")
print(f"   Glicose inválida  : {((df['glicose'] < 70) | (df['glicose'] > 220)).sum()}")
print(f"   Classes presentes : {sorted(df['risco'].unique())}")


# ══════════════════════════════════════════════════════════════════
# PASSO 6 — EXPORTAÇÃO PARA CSV
#
# index=False  → não salva o índice do DataFrame como coluna
# encoding     → UTF-8 para suporte a caracteres especiais (acentos)
# ══════════════════════════════════════════════════════════════════
df.to_csv("pacientes.csv", index=False, encoding="utf-8")

print(f"\n{'─' * 65}")
print("✅ Dataset exportado com sucesso → pacientes.csv")
print("=" * 65)
