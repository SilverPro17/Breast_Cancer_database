from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

try:
    data = load_breast_cancer()
    X = data.data
    y = data.target
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Forma de X_train: {X_train.shape}, Forma de X_test: {X_test.shape}")
print(f"Forma de y_train: {y_train.shape}, Forma de y_test: {y_test.shape}")

# 2. Inicializar e treinar o Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

print("\nDecision Tree Classifier treinado com sucesso.")

# 3. Fazer previsões no conjunto de teste
y_pred_dt = clf.predict(X_test)

# 4. Avaliar o classificador
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt, target_names=data.target_names)

print(f"\nAvaliação do Decision Tree Classifier (Supervisionado):")
print(f"Acurácia: {accuracy_dt:.4f}")
print(f"\nMatriz de Confusão:")
print(f"Classes (linhas/colunas): {data.target_names[0]} (0), {data.target_names[1]} (1)")
print(conf_matrix_dt)
print(f"\nRelatório de Classificação:")
print(class_report_dt)

np.save("C:/Users/rodri/Documents/Escola/4_ANO/2º SEMESTRE/Seminário(Lau)/TPC1/y_test_dt.npy", y_test)
np.save("C:/Users/rodri/Documents/Escola/4_ANO/2º SEMESTRE/Seminário(Lau)/TPC1/y_pred_dt.npy", y_pred_dt)

print("\nAvaliação do Decision Tree concluída. Resultados e previsões guardados (opcionalmente).")

