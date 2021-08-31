import numpy as np
import pandas as pd
import re
import operator
from time import time
from datetime import datetime, timedelta
from random import randrange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score,precision_score,recall_score, make_scorer

# ------------------------------------------------------------------------------------------
#    Alunos:    William Felipe Tsubota      - 2017.1904.056-7
#               Matheus Albuquerque de melo - 2016.1907.030-8
# ------------------------------------------------------------------------------------------

import warnings
# Removendo avisos que aparecem na tela na execução do algoritmo
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Arquivo contendo o dataset
DATASET = 'dataset/dataset.data'

#Arquivo contendo as respostas
ANSWER = 'dataset/answers.data'

#Arquivo contendo o etiquetador de entidades
ENTITY = 'dataset/entitylabel.data'

#Arquivo contendo os preços

PRICE = 'dataset/prices.data'

# Coluna que contem a classe do arquivo dataset
CLASS = 1

# Simplificando as classes
mapping = {'festas' : 'festa', 'associacao' : 'associar', 'produtos' : 'produto',
 'saudacao' : 'saudacao','comprar': 'comprar','despedida' : 'despedida'}

# Mapeando plurais para ficar alinhado com o dataset de precos, que esta no singular
mappingEntity = {'moletons' : 'moletom', 'canecas' : 'caneca', 'colleges' : 'college', 
'tirantes': 'tirante', 'blusas' : 'blusa',  'camisetas' : 'camiseta', 
'camisas' : 'camisa', 'abadas' : 'abada', 'abadá' : 'abada', 'abadás' : 'abada',
'blusinhas' : 'blusinha', 'chinelos' : 'chinelo', 'jaquetas' : 'jaqueta', 'casacos' : 'casaco',
'bandoleiras' : 'bandoleira', 'óculos' : 'oculos', 'três' : 'tres', 'catorze' : 'quatorze', 'mítica': 'mitica'}

#preparando as metricas para usar no grid_search

scorers = {
		'precision_score': make_scorer(precision_score, average='macro'),
		'accuracy_score': make_scorer(accuracy_score),
		'recall_score': make_scorer(recall_score, average='macro')
	}

# Funcao que le o dataset de treinamento
def reader():
	data = pd.read_csv(DATASET, header=None, sep='-')    
	print(data)
	# Lemos diretamente da matriz
	x,y = data.iloc[:, 0:CLASS].values, data.iloc[:, CLASS].values
	# Transformamos os nomes em outras strings:
	for i in range(len(y)):
		y[i] = mapping[y[i]]
	# Transformando em np array, retirando elementos array dentro de array, e transformando de np array para lista simples novamente
	x = np.array(x).ravel().tolist()
	y = y.astype(str, casting='unsafe')
	return x, y

# Funcao que le o dataset das respostas
def readerAnswer():
	data = pd.read_csv(ANSWER, header=None, sep='-')    
	print(data)
	answerDic = {}

	for answer, classification in data.iloc[:,[0,1]].values:
		if mapping[classification] not in answerDic.keys():
			answerDic[mapping[classification]] = []
		answerDic[mapping[classification]].append(answer)
		# print(answer,mapping[classification])
	# print(answerDic)
	return answerDic

# Funcao que le o dataset para reconhecer as entidades da frase do usuario posteriormente
def readerEntity():
	data = pd.read_csv(ENTITY, header=None, sep='-')    
	print(data)
	entityDic = {}

	for entity, classification in data.iloc[:,[0,1]].values:
		if classification not in entityDic.keys():
			entityDic[classification] = []
		entityDic[classification].append(entity)
		# print(entity,classification)
	# print(entityDic)
	return entityDic

# Funcao que le o dataset de preco de cada item
def readerPrice():
	data = pd.read_csv(PRICE, header=None, sep='-')    
	print(data)
	priceDic = {}

	for price, classification in data.iloc[:,[0,1]].values:
		priceDic[classification] = price
		# print(price,classification)
	# print(priceDic)
	return priceDic

# Método grid_search, recebe os respectivos parâmetros e retorna a variável com o search
def grid_Search(param_grid, estimator, scorers, my_cv):

	  clf = GridSearchCV(
				estimator=estimator,
				param_grid=param_grid,
				scoring=scorers,
				refit='accuracy_score',
				verbose=0,
				cv=my_cv,
				n_jobs=1,
				return_train_score=True)
	  return clf

def nb(x,y):
	my_cv = LeaveOneOut()

	param_grid ={'alpha': [0, 1/3, 2/3, 1],
			 'fit_prior': [True,False]}

	nb = BernoulliNB()


   

	clf = grid_Search(param_grid,nb,scorers,my_cv)

	# Ao passar x (frase) e y (classe) é de fato que o grid search é executado e ele já treina automaticamente, pois cv = leave in out ()
	clf.fit(x, y)
	# Passando o melhor parametro para o nb
	nb = clf.best_estimator_

	loo = LeaveOneOut()
	loo.get_n_splits(x)

	# Lista para guardar as classes preditas de cada frase usada como teste
	y_predict = []
	# For do leave one out com o melhor parâmetro achado pelo grid search
	for train_index, test_index in loo.split(x):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			nb.fit(x_train, y_train)
			y_predict.append(nb.predict(x_test))

   

	return nb

# Funcao para reconhecer as entidades em uma frase que o usuario passou
# Ela recebe um dicionario de entidades e a frase do usuario
def reconhece_entidades(entities, text):
	text = re.sub('[;,.:_!?]', ' ', text, flags = re.IGNORECASE)
	# Inicializacao do dicionario das entidades reconhecidas na frase do usuario
	entityRecognizedDic = {}
	# Este dicionario abaixo mescla produto e ingresso em uma mesma chave, pois isso facilita quando vamos casar
	# Exemplo: para a frase quero dois moletons tres ingressos para a computaria, um tirante, duas canecas
	# e quatro ingresos para a mitica
	# para o entityRecognizedDic ele teria a seguinte estrutura: {'numero': ['dois', 'tres', 'um', 'duas', 'quatro'], 
	# 'produto': ['moletons', 'tirante', 'canecas'], 'ingresso': ['computaria', 'mitica']}
	# Enquanto o entityRecognizedDicInOrder teria {'numero': ['dois', 'tres', 'um', 'duas', 'quatro'], 
	# 'produtoingresso': ['moletons', 'computaria', 'tirante', 'canecas', 'mitica']}
	# Ou seja está na ordem do pedido, enquanto no entityRecognizedDic não saberíamos se ele pediu um ingresso ou produto primeiro
	entityRecognizedDicInOrder = {}
	entityRecognizedDicInOrder['produtoingresso'] = []
	entityRecognizedDicInOrder['numero'] = []
	for word in text.split():
		# Variável auxiliar que guarda a palavra em minúscula, a partir daqui trabalhamos para reconhecer tudo
		# Em minusculo, pois no nossod ata set so temos dados em minusculo
		wordLower = word.lower()
		for label in entities.keys():
			# Neste if estamos transformando uma possivel palavra em plural em singular com o mapeamento dos produtos
			if wordLower in mappingEntity.keys():
				wordLower = mappingEntity[wordLower]
			# Neste if verificamos se a palavra eh uma entidade daquela label
			if wordLower in entities[label]:
				# Caso seja verificamos se ela ja tem uma lista inicializada
				if label not in entityRecognizedDic.keys():
					entityRecognizedDic[label] = []
				# Entao inserimos na lista
				entityRecognizedDic[label].append(wordLower)
				# Se nao eh numero, so pode ser produto ou ingresso, entao adicionamos na lista em ordem
				if(label != 'numero'):
					entityRecognizedDicInOrder['produtoingresso'].append(wordLower)
				# Esse else quer dizer que label eh um numero, entao somente adicionamos a palavra em minusculo na posicao label
				else:
					entityRecognizedDicInOrder[label].append(wordLower)
				break;
	# Retornamos os dois dicionarios
	return entityRecognizedDic, entityRecognizedDicInOrder

# Função usada para saudação específica dependendo do horário, ex: bom dia, boa tarde, boa noite, boa madrugada
def hour():
	if(datetime.now().hour*60 + datetime.now().minute >= int(0)*60 + int(00) 
		and datetime.now().hour*60 + datetime.now().minute < int(6)*60 + int(00)):
		print('Boa madrugada! Aqui é da AAACOMP, como posso te ajudar?')
	
	elif(datetime.now().hour*60 + datetime.now().minute >= int(6)*60 + int(00) and 
	   datetime.now().hour*60 + datetime.now().minute < int(12)*60 + int(00)):
		print('Bom dia! Aqui é da AAACOMP, como posso te ajudar?')

	elif(datetime.now().hour*60 + datetime.now().minute >= int(12)*60 + int(00) and 
	   datetime.now().hour*60 + datetime.now().minute < int(18)*60 + int(00)):
		print('Boa tarde! Aqui é da AAACOMP, como posso te ajudar?')

	else:
		print('Boa noite! Aqui é da AAACOMP, como posso te ajudar?')

# Função usada para calcular os reconhecidos pelo reconhecedor de entidades
# Basicamente ele casa a quantidade dos numerais com os itens
def calculateItems(entitiesInTextInOrder, prices, entities):
	# O usuario digitou pelo menos um item e pelo menos um numero
	if(len(entitiesInTextInOrder['numero']) != 0):
		# A quantidade de items não condizeu com a quantidade de numerais reconhecido
		if(len(entitiesInTextInOrder['produtoingresso'] ) != len(entitiesInTextInOrder['numero'])):
			print('Desculpe-me, mas faltou items e/ou quantidades no seu pedido')
		else:
			# Se o usuario digitou numerais e items na mesma quantidade, entao poderei gerar a nota fiscal
			totalpedido = 0.0
			print('\nnota fiscal:\n')
			# For simultaneo de uma lista que pode conter produto/ingresso e sua respectiva quantidade
			for item, number in zip(entitiesInTextInOrder['produtoingresso'], entitiesInTextInOrder['numero']):
				price = 0
				quantidade = 0
				index = entities['numero'].index(number)
				# Os numeros 1 ate 20, estao com o o indice de 0 ate 19 no dataset entitylabel
				if (index < 20):
					quantidade = index + 1
					price = (index + 1) * float(prices[item])
				# Os numeros por extenso de um ate vinte estao no indice 20 ate o 39, logo eh so tirar 19 do indice
				elif (index < 40):
					quantidade = index - 19
					price = (index - 19) * float(prices[item])
				# O numero eh uma ou duas, que estao na posicao 40 e 41, respectivamente
				else:
					quantidade = index - 39
					price = (index - 39) * float(prices[item])
				print('item(s):',item,';preco unitario :',float(prices[item]),';quantidade:',quantidade,'unidade(s) ;preco total R$:',str(round(price,2)),'reais')
				totalpedido += round(price,2) 
			print('\n')
			print('O seu pedido deu um total de',str(round(totalpedido,2)), 'reais, clique aqui e pague viva paypal')

	# Else: usuario nao digitou corretamente a relacao item/produto
	else:
		print('Desculpe-me, mas eu nao entendi o seu pedido, voce poderia, por favor, verifica-lo, corrigi-lo e manda-lo de novo?')


def response(preditor, vectorizer, answers, entities, prices):
	print("O preditor escolhido foi: ", preditor)
	while(True):
		text = (input("[*] Digite seu texto \n"))
		inst = vectorizer.transform([text])

		probability = pd.DataFrame(preditor.predict_proba(inst), columns=preditor.classes_)
		predict = preditor.predict(inst)

		print('Intencao detectada:' , predict , '\n')
		print(probability)

		
		

		# Estou aleatoriezando uma possivel resposta no dataset de respostas e colocando o indice em uma variavel
		positionAnswer = randrange(0, (len(answers[predict[0]])))
		entitiesInText, entitiesInTextInOrder = reconhece_entidades(entities, text)
		
		if(predict == 'comprar'):
			# A intenção foi de comprar
			# Ver se ele passou quantidade do item e o item
			# Se sim, então calcula e manda pra ele
			# Se não, pergunta o item (se esse faltou) e/ou a quantidade (se esse faltou também)
			calculateItems(entitiesInTextInOrder, prices, entities)
			# print('tenho que calcular o preco')
		elif(predict == 'saudacao'):
			# No dataset de respostas, ha 7 respostas para saudacao, no qual 4 delas eh referente a saudacao especifica
			# de horario, portanto se cair no indice de alguma delas, eu pego a hora atual e respondo de acordo com
			# o horario
			if(positionAnswer <= 3):
				hour()
			# Se cair nos outros 3 indices, entao eu dou uma resposta aleatoria dessas outras 3
			else:
				print(answers[predict[0]][positionAnswer])
		# Else if que detecta que o usuario quer ver o produto, aqui mostramos todos os produtos para ousuario
		elif(predict == 'produto'):
			print('Estes sao os precos dos nossos produtos:\n')
			for produto in entities['produto']:
				print(produto,'-',prices[produto], 'reais')
		# Similar ao else if de cima, so que aqui mostramos as festa
		elif(predict == 'festa'):
			print('Estes sao os precos dos ingressos das nossas festas:\n')
			for festa in entities['ingresso']:
				print(festa,'-',prices[festa], 'reais')
		# Se nao eh nenhuma predicao de cima, entao pegamos uma resposta aleatoria
		else:
			# Retornamos a resposta ao usuario
			print(answers[predict[0]][positionAnswer])
		
		# Printamos as entidades na tela do terminal
		print(entitiesInText)

		


def main():
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,strip_accents='unicode')
	corpus, y = reader()
	answers = readerAnswer()
	entities = readerEntity()
	prices = readerPrice()
	y = np.array(y)
	x = vectorizer.fit_transform(corpus)

	naive = nb(x,y)

	response(naive,vectorizer,answers, entities, prices)

	

if __name__ == "__main__":
	main()