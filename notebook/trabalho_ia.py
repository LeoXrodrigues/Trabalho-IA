# =============================================================================
# Disciplina de Inteligência Artificial | Professor Munif | Unicesumar 2026
# Trabalho Final — Comparação de Modelos KNN vs Random Forest
# Dataset: HIIT vs Hypertrophy Workout Gains
# =============================================================================

# %% [1] IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})

print("=" * 60)
print("  TRABALHO FINAL — INTELIGÊNCIA ARTIFICIAL")
print("  Disciplina: IA | Professor Munif | Unicesumar 2026")
print("=" * 60)


# %% [2] CARREGAR E EXPLORAR O DATASET
print("\n[PASSO 1] Carregando o dataset...")
df = pd.read_csv('../dataset/hiit_vs_hypertrophy_gains.csv')

print(f"\nShape do dataset: {df.shape}")
print(f"Registros: {df.shape[0]} | Colunas: {df.shape[1]}")
print("\nPrimeiras linhas:")
print(df.head())

print("\nTipos de dados:")
print(df.dtypes)

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nVerificação de valores nulos:")
print(df.isnull().sum())
print(f"\nTotal de nulos: {df.isnull().sum().sum()}")

print("\nDistribuição da variável alvo (Group):")
print(df['Group'].value_counts())


# %% [3] GRÁFICO — DISTRIBUIÇÃO DA VARIÁVEL ALVO
PALETTE = ['#2196F3', '#FF5722', '#4CAF50']
fig, ax = plt.subplots(figsize=(8, 5))
counts = df['Group'].value_counts()
bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_title('Distribuição da Variável Alvo (Group)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Tipo de Treino')
ax.set_ylabel('Quantidade de Registros')
ax.set_ylim(0, counts.max() + 15)
plt.tight_layout()
plt.savefig('../graphs/distribuicao_alvo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: graphs/distribuicao_alvo.png")


# %% [4] PRÉ-PROCESSAMENTO
print("\n[PASSO 2] Pré-processamento dos dados...")

# 4.1 Remover ID (não contribui como feature)
df = df.drop(columns=['Participant_ID'])

# 4.2 Codificação das variáveis categóricas
le_gender = LabelEncoder()
le_diet   = LabelEncoder()
le_target = LabelEncoder()

df['Gender']            = le_gender.fit_transform(df['Gender'])
df['Dietary_Condition'] = le_diet.fit_transform(df['Dietary_Condition'])

print(f"\nGêneros codificados: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
print(f"Dieta codificada: {dict(zip(le_diet.classes_, le_diet.transform(le_diet.classes_)))}")

# 4.3 Separar X e y
X = df.drop(columns=['Group'])
y = le_target.fit_transform(df['Group'])

print(f"\nClasses da variável alvo: {le_target.classes_}")
print(f"Codificação: {dict(zip(le_target.classes_, range(len(le_target.classes_))))}")
print(f"\nShape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# 4.4 Normalização com StandardScaler
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n✅ Features normalizadas com StandardScaler")

# 4.5 Holdout — 70% treino / 30% teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y)

print(f"\n[HOLDOUT] Divisão dos dados:")
print(f"  Treino : {X_train.shape[0]} registros (70%)")
print(f"  Teste  : {X_test.shape[0]} registros (30%)")


# %% [5] TREINAR KNN
print("\n[PASSO 3] Treinando KNN (K=5)...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("✅ KNN treinado!")


# %% [6] TREINAR RANDOM FOREST
print("\n[PASSO 4] Treinando Random Forest (100 árvores)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("✅ Random Forest treinado!")


# %% [7] AVALIAÇÃO DOS MODELOS
classes = le_target.classes_

def avaliar_modelo(y_true, y_pred, nome, classes):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec  = recall_score(y_true, y_pred, average='weighted')
    f1   = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{'='*50}")
    print(f"  MODELO: {nome}")
    print(f"{'='*50}")
    print(f"  Acurácia : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precisão : {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\nRelatório completo:")
    print(classification_report(y_true, y_pred, target_names=classes))
    return acc, prec, rec, f1

print("\n[PASSO 5] Avaliando modelos...")
acc_knn, prec_knn, rec_knn, f1_knn = avaliar_modelo(y_test, y_pred_knn, "KNN", classes)
acc_rf,  prec_rf,  rec_rf,  f1_rf  = avaliar_modelo(y_test, y_pred_rf,  "Random Forest", classes)


# %% [8] MATRIZ DE CONFUSÃO — KNN
fig, ax = plt.subplots(figsize=(7, 6))
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={"size": 14})
ax.set_title('Matriz de Confusão — KNN', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Predito', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
plt.tight_layout()
plt.savefig('../graphs/matriz_confusao_knn.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: graphs/matriz_confusao_knn.png")


# %% [9] MATRIZ DE CONFUSÃO — RANDOM FOREST
fig, ax = plt.subplots(figsize=(7, 6))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=classes, yticklabels=classes, ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={"size": 14})
ax.set_title('Matriz de Confusão — Random Forest', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Predito', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
plt.tight_layout()
plt.savefig('../graphs/matriz_confusao_random_forest.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: graphs/matriz_confusao_random_forest.png")


# %% [10] COMPARAÇÃO DE ACURÁCIA
fig, ax = plt.subplots(figsize=(7, 5))
modelos   = ['KNN', 'Random Forest']
acuracias = [acc_knn, acc_rf]
cores     = ['#2196F3', '#FF5722']
bars = ax.bar(modelos, acuracias, color=cores, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, acuracias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_title('Comparação de Acurácia', fontsize=15, fontweight='bold', pad=12)
ax.set_ylabel('Acurácia')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
ax.legend()
plt.tight_layout()
plt.savefig('../graphs/comparacao_acuracia.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: graphs/comparacao_acuracia.png")


# %% [11] COMPARAÇÃO DE F1-SCORE
fig, ax = plt.subplots(figsize=(7, 5))
f1s = [f1_knn, f1_rf]
bars = ax.bar(modelos, f1s, color=cores, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_title('Comparação de F1-Score (Weighted)', fontsize=15, fontweight='bold', pad=12)
ax.set_ylabel('F1-Score')
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('../graphs/comparacao_f1_score.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: graphs/comparacao_f1_score.png")


# %% [12] COMPARAÇÃO COMPLETA DE MÉTRICAS
fig, ax = plt.subplots(figsize=(9, 5))
metricas_nomes = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
vals_knn = [acc_knn, prec_knn, rec_knn, f1_knn]
vals_rf  = [acc_rf,  prec_rf,  rec_rf,  f1_rf]
x_pos = np.arange(len(metricas_nomes))
width = 0.35
bars1 = ax.bar(x_pos - width/2, vals_knn, width, label='KNN', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x_pos + width/2, vals_rf,  width, label='Random Forest', color='#FF5722', edgecolor='white')
for bar, val in zip(list(bars1)+list(bars2), vals_knn+vals_rf):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metricas_nomes, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title('Comparação Completa de Métricas', fontsize=15, fontweight='bold', pad=12)
ax.set_ylabel('Valor')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('../graphs/comparacao_metricas.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: graphs/comparacao_metricas.png")


# %% [13] SALVAR MODELOS TREINADOS
joblib.dump(knn,    '../models/knn_model.pkl')
joblib.dump(rf,     '../models/random_forest_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
print("\n✅ Modelos salvos em models/")


# %% [14] RESUMO FINAL
print("\n" + "=" * 60)
print("  RESUMO FINAL DOS RESULTADOS")
print("=" * 60)
print(f"\n  {'Métrica':<15} {'KNN':>12} {'Random Forest':>15}")
print(f"  {'-'*42}")
print(f"  {'Acurácia':<15} {acc_knn:>12.4f} {acc_rf:>15.4f}")
print(f"  {'Precisão':<15} {prec_knn:>12.4f} {prec_rf:>15.4f}")
print(f"  {'Recall':<15} {rec_knn:>12.4f} {rec_rf:>15.4f}")
print(f"  {'F1-Score':<15} {f1_knn:>12.4f} {f1_rf:>15.4f}")
print(f"\n  ✅ Melhor modelo: Random Forest")
print(f"  ✅ Diferença de acurácia: {(acc_rf - acc_knn)*100:.2f} pontos percentuais")
