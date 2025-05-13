# Relatório Comparativo: Clustering Não Supervisionado vs. Classificação Supervisionada no Dataset de Cancro da Mama

## Introdução

Este relatório apresenta uma análise comparativa entre uma abordagem de aprendizagem não supervisionada (clustering com K-Means) e uma abordagem de aprendizagem supervisionada (classificação com Decision Tree Classifier) aplicadas ao dataset de cancro da mama. O objetivo principal do exercício foi investigar se seria possível identificar agrupamentos correspondentes às classes de tumores (maligno ou benigno) utilizando apenas as características dos dados, sem recorrer aos rótulos verdadeiros (aprendizagem não supervisionada), e, subsequentemente, quantificar a melhoria de desempenho ao utilizar um classificador treinado com esses rótulos (aprendizagem supervisionada).

## Abordagem Não Supervisionada: Clustering com K-Means

Na primeira fase, os dados do dataset de cancro da mama foram processados utilizando o algoritmo K-Means, com o objetivo de os agrupar em dois clusters, na esperança de que estes correspondessem às categorias de tumores malignos e benignos. Antes da aplicação do K-Means, as características dos dados (features) foram normalizadas utilizando `StandardScaler` para garantir que todas as features contribuíssem de forma equitativa para o cálculo das distâncias, dado que o K-Means é sensível à escala dos dados.

Os resultados do clustering foram avaliados comparando os rótulos dos clusters atribuídos a cada amostra com os rótulos verdadeiros (maligno/benigno) do dataset. As métricas obtidas foram as seguintes: o Adjusted Rand Index (ARI) alcançou um valor de 0.6765, enquanto o Normalized Mutual Information (NMI) foi de 0.5620. A homogeneidade, que mede se cada cluster contém apenas membros de uma única classe, foi de 0.5510, e a completude, que mede se todos os membros de uma dada classe são atribuídos ao mesmo cluster, foi de 0.5735. O V-measure, que é a média harmónica da homogeneidade e completude, resultou em 0.5620.

A matriz de contingência, que cruza as classes verdadeiras com os clusters formados, revelou que o Cluster 0 continha 175 amostras malignas e 13 benignas, enquanto o Cluster 1 continha 37 amostras malignas e 344 benignas. Com base nesta matriz, foi calculada uma "pseudo-acurácia", que tenta alinhar os clusters com as classes reais para obter uma medida de correspondência. Esta pseudo-acurácia atingiu 0.9121. É importante notar que esta é uma medida simplificada, e métricas como ARI e NMI são mais robustas para a avaliação de clustering, pois não dependem de um alinhamento manual dos rótulos dos clusters.

Estes resultados indicam que o K-Means conseguiu, até certo ponto, identificar uma estrutura nos dados que se correlaciona com as classes reais, mas com uma separação imperfeita, como evidenciado pelos valores de ARI e NMI, e pela presença de amostras de ambas as classes em cada cluster.

## Abordagem Supervisionada: Classificação com Decision Tree Classifier

Na segunda fase, foi treinado um classificador supervisionado, especificamente um Decision Tree Classifier, utilizando as mesmas características do dataset, mas desta vez fornecendo também os rótulos verdadeiros (maligno/benigno) durante o processo de treino. Os dados foram divididos em conjuntos de treino (70%) e teste (30%) para avaliar a capacidade de generalização do modelo em dados não vistos.

O Decision Tree Classifier, após o treino, foi avaliado no conjunto de teste. A acurácia obtida foi de 0.9181. A matriz de confusão para o modelo supervisionado mostrou que, das 64 amostras malignas no conjunto de teste, 57 foram corretamente classificadas como malignas e 7 foram incorretamente classificadas como benignas. Das 107 amostras benignas, 100 foram corretamente classificadas como benignas e 7 foram incorretamente classificadas como malignas.

O relatório de classificação detalhado forneceu as métricas de precisão, recall e F1-score para cada classe. Para a classe "malignant", a precisão foi de 0.89, o recall de 0.89 e o F1-score de 0.89. Para a classe "benign", a precisão foi de 0.93, o recall de 0.93 e o F1-score de 0.93. Estes valores indicam um bom desempenho do classificador supervisionado na distinção entre as duas classes.

## Comparação e Conclusão

Ao comparar as duas abordagens, observamos que o Decision Tree Classifier (aprendizagem supervisionada) alcançou uma acurácia de 0.9181 no conjunto de teste. A abordagem de clustering com K-Means (aprendizagem não supervisionada), quando avaliada através da pseudo-acurácia após alinhamento dos clusters, obteve um valor de 0.9121. Embora estes valores de acurácia pareçam próximos, é crucial considerar a natureza das métricas. A acurácia do modelo supervisionado é uma medida direta da sua capacidade preditiva em dados rotulados, enquanto a pseudo-acurácia do clustering é uma estimativa baseada na melhor correspondência possível entre os clusters formados e as classes preexistentes.

As métricas mais formais de avaliação de clustering, como o ARI (0.6765) e o NMI (0.5620), sugerem que, embora o K-Means tenha encontrado alguma estrutura relevante, a qualidade do agrupamento em termos de correspondência com as classes reais é significativamente inferior à capacidade de um modelo treinado explicitamente para distinguir essas classes.

Em conclusão, o exercício demonstra que, para a tarefa de classificar tumores como malignos ou benignos no dataset de cancro da mama, a aprendizagem supervisionada, utilizando um Decision Tree Classifier, oferece um desempenho superior e mais fiável em comparação com a tentativa de inferir estas classes através de clustering não supervisionado com K-Means. Embora o clustering possa revelar padrões interessantes nos dados na ausência de rótulos, a disponibilidade de rótulos permite treinar modelos que aprendem diretamente a mapear as características para as classes desejadas, resultando numa maior precisão e robustez na classificação.

