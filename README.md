# AUSTRALOPITEKUS-APHARENSIS-IA

Nomes:

William Felipe Tsubota - 2017.1904.056-7
Matheus Albuquerque de melo - 2016.1907.030-8

Preditores (estimadores) utilizados:

* Knn
* Regressão logística
* Naive bayers bernoulli
* Naive bayers multinominal
* Árvore de decisão
* Redes neurais

Intenções (classes) utilizadas:

Produtos - utilizado para ver os produtos
Comprar - utilizado para comprar produtos e ingressos de festas
Festas - utilizado para ver festas
Associar - informações sobre associação
Saudação - saudação
Despedida - despedida

Relatório:

* Primeiramente lemos do nosso dataset com a função "reader()", dentro dela usamos o módulo pandas para ler os dados do dataset, nosso separador de frases (x) e intenção (y) é o caractere "-".
* Vetorizamos a frase com o vectorizer para que os algoritmos do sklearn entendam.
* Métodos dos algoritmos:
	** Passamos para cada método dos algoritmos o array de frases "x" e o array de clases "y"
	** Utilizamos o grid search para achar a melhor combinação parâmetros dentre os parâmetros que a gente listou no param_grid (Testamos algumas combinações de parâmetros para listar no param_grid), no grid search utilizamos o cross validation leave one out (também conhecido como KFold(n_splits=n))
	** Após rodar o grid search é gerado um relatório referente aos parâmetros e valores da acurácia, este é gravado em 'nome do preditor'.gs_results.csv
	** Após rodar o grid search pegamos a melhor combinação de parâmetros (o que tem a maior acurácia), criamos um novo preditor com esses parâmetros e fazemos um for para treinar o algoritmo com o método do leave one out, damos um fit no treino e depois predict no teste, e guardamos esse predict em um vetor, após rodar o for do leave one out comparamos a classe y real com o y predito para obter os valores de acurácia, precisão e recall.
	** Após o for do leave one out salvamos esse resultado (acurácia, precisão e recall) em 'nome do preditor'.best_result.csv, salvamos também os parâmetros do preditor com a melhor combinação de parâmetros
* Retornamos o preditor treinado de cada algoritmo e sua respectiva acurácia
* Escolhemos o melhor preditor para utilizar de fato na nossa predição das frases (basicamente um while true, no qual o usuário digita as frases e o preditor indica a intenção)
* Como sugerido pelo monitor, utilizamos como critério de desempate, a maior acurácia dentre as melhores combinações dos preditores gerados pelo grid search retornados para a main (simplificando: pra cada algoritmo vemos qual combinação de parâmetros tem a melhor acurácia, e depois de pegar a melhor combinação de parâmetros desses algoritmos, vemos qual algoritmo tem a maior acurácia e utilizamos este).

Obs: Devido a demora de execução do algoritmo de redes neurais, este estará comentado na main, para utilizá-lo basta descomentá-lo
