import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

try:
    y_true = np.load("C:/Users/rodri/Documents/Escola/4_ANO/2º SEMESTRE/Seminário(Lau)/TPC1/y_true_breast_cancer.npy")
    cluster_labels = np.load("C:/Users/rodri/Documents/Escola/4_ANO/2º SEMESTRE/Seminário(Lau)/TPC1/cluster_labels.npy")
except FileNotFoundError:
    print("Erro: Ficheiros y_true_breast_cancer.npy ou cluster_labels.npy não encontrados. Execute os passos anteriores primeiro.")
    exit()
    
data = load_breast_cancer()
target_names = data.target_names

print(f"Forma de y_true: {y_true.shape}")
print(f"Forma de cluster_labels: {cluster_labels.shape}")

ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)
homogeneity = homogeneity_score(y_true, cluster_labels)
completeness = completeness_score(y_true, cluster_labels)
v_measure = v_measure_score(y_true, cluster_labels)

print(f"\nAvaliação da Correspondência entre Clusters K-Means e Classes Reais:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Homogeneidade: {homogeneity:.4f}")
print(f"Completude: {completeness:.4f}")
print(f"V-measure: {v_measure:.4f}")

contingency_mat = confusion_matrix(y_true, cluster_labels)
print(f"\nMatriz de Contingência (Classes Reais vs. Clusters K-Means):")
print(f"Classes Reais (linhas): {target_names[0]} (0), {target_names[1]} (1)")
print(f"Clusters K-Means (colunas): Cluster 0, Cluster 1")
print(contingency_mat)

# Interpretação da Matriz de Contingência:
# Se cluster_labels[i] == 0 e y_true[i] == 0 (maligno), conta para contingency_mat[0,0]
# Se cluster_labels[i] == 1 e y_true[i] == 0 (maligno), conta para contingency_mat[0,1]
# Se cluster_labels[i] == 0 e y_true[i] == 1 (benigno), conta para contingency_mat[1,0]
# Se cluster_labels[i] == 1 e y_true[i] == 1 (benigno), conta para contingency_mat[1,1]


# Contagem de y_true = 0 (maligno) em cada cluster
malignant_in_cluster0 = contingency_mat[0,0]
malignant_in_cluster1 = contingency_mat[0,1]

# Contagem de y_true = 1 (benigno) em cada cluster
benign_in_cluster0 = contingency_mat[1,0]
benign_in_cluster1 = contingency_mat[1,1]

print(f"\nAnálise da Matriz de Contingência:")
print(f"Classe '{target_names[0]}' (0):")
print(f"  - No Cluster 0: {malignant_in_cluster0}")
print(f"  - No Cluster 1: {malignant_in_cluster1}")
print(f"Classe '{target_names[1]}' (1):")
print(f"  - No Cluster 0: {benign_in_cluster0}")
print(f"  - No Cluster 1: {benign_in_cluster1}")

accuracy_case1 = (contingency_mat[0,0] + contingency_mat[1,1]) / np.sum(contingency_mat)
accuracy_case2 = (contingency_mat[0,1] + contingency_mat[1,0]) / np.sum(contingency_mat)

pseudo_accuracy = max(accuracy_case1, accuracy_case2)
print(f"\nPseudo-Acurácia (após alinhar clusters com classes): {pseudo_accuracy:.4f}")
print("Nota: Esta 'pseudo-acurácia' é uma medida simplificada. Métricas como ARI e NMI são mais robustas para avaliação de clustering.")

