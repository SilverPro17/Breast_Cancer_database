from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer 
import numpy as np

data = load_breast_cancer()
X = data.data
y_true = data.target 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)

print(f"Rótulos dos clusters atribuídos (primeiros 20): {cluster_labels[:20]}")
print(f"Centros dos clusters encontrados (shape): {kmeans.cluster_centers_.shape}")

np.save('C:/Users/rodri/Documents/Escola/4_ANO/2º SEMESTRE/Seminário(Lau)/TPC1/cluster_labels.npy', cluster_labels)
np.save('C:/Users/rodri/Documents/Escola/4_ANO/2º SEMESTRE/Seminário(Lau)/TPC1/y_true_breast_cancer.npy', y_true) # Guardar y_true também para a comparação

print("\nClustering K-Means concluído. Os rótulos dos clusters foram guardados em cluster_labels.npy e os rótulos verdadeiros em y_true_breast_cancer.npy.")

