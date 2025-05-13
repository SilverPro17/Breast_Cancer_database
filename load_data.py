from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

# Imprimir informações básicas sobre o dataset
print(f"Número de amostras: {X.shape[0]}")
print(f"Número de features: {X.shape[1]}")
print(f"Nomes das features: {data.feature_names}")
print(f"Nomes das classes (target): {data.target_names}")
dist_values = {data.target_names[0]: sum(y == 0), data.target_names[1]: sum(y == 1)}
print(f"Distribuição das classes: {dist_values}")
print("\nPrimeiras 5 amostras de X:")
print(X[:5])
print("\nPrimeiras 5 amostras de y:")
print(y[:5])

print("\nDataset carregado com sucesso. X e y estão disponíveis.")

