import os
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from itertools import combinations

class Aluno:
	def __init__(self, idd, dados):
		dados = dados.split("\t")
		self.id = idd
		self.age = dados[1]
		self.gender = True if dados[2] == '\"female\"' else False
		self.only = True if dados[3] == '\"yes\"' else False
		self.education = dados[4]
		self.infos = []
		self.result = -1

		
	def addValues(self, infos):
		self.infos.append(np.array(infos)/5)


	def getInfos(self, files):
		toReturn = self.infos[files[0]]
		for f in files[1:]:
			toReturn = np.concatenate((toReturn, self.infos[f]))
		return toReturn
	
	def __repr__(self):
		return "id: " + str(self.id) + " age: " + self.age + " gender: " + self.gender + " only: " + self.only + " education: " + self.education


class PorFile:
	def __init__(self, dados	):
		self.dados = np.matrix(dados)/5
		



caminho = "data\\"


files = ['HobbiesAndInterests_Vars.txt', 'MusicAndMovies_Vars.txt', 'Personality_Vars.txt', 'Phobias_Vars.txt', 'SpendingHabits_Vars.txt']


demographics = open(caminho +  'SocioDemographic_Vars.txt').read().split("\n")[1:]


atributos = []

alunos = [Aluno(i, demographics[i]) for i in range(1010)]
matrizes = []


homens = []
mulheres = []
filhos_unico = []
tem_irmão = []

for idd, a in enumerate(alunos):
	if a.gender:
		mulheres.append(idd)
	else:
		homens.append(idd)
	if a.only:
		filhos_unico.append(idd)
	else:
		tem_irmão.append(idd)

print(len(homens))
print(len(mulheres))

homens = np.array(homens)
mulheres = np.array(mulheres)
filhos_unico = np.array(filhos_unico)
tem_irmão = np.array(tem_irmão)



centers = 6
file_var = [3]
homens_mulher = 1

kmeans = KMeans(n_clusters=centers, init='k-means++', max_iter=300, n_init=10, random_state=0)

#for d in dados:
#	print(d)



for f in files:
	txt = open(caminho + f).read()
	linhas = txt.split("\n")
	atributos.append(linhas.pop(0).split("\t"))
	m = []

	for idd,  l in enumerate(linhas[:-1]):
		values = list(map(int, l.split("\t")[1:]))
		alunos[idd].addValues(values)
		m.append(values)
	matrizes.append(PorFile(m))


to_k_means = matrizes[file_var[0]].dados
for f in file_var[1:]:
	to_k_means = np.concatenate((to_k_means, matrizes[f].dados), axis=1)


if homens_mulher == 0:
	to_k_means = np.delete(to_k_means, homens, axis = 0)
elif homens_mulher == 1:
	to_k_means = np.delete(to_k_means, mulheres, axis = 0)

kmeans.fit(to_k_means)


testes = np.zeros(centers)
total = 0

num_testes = len(to_k_means)
for line in to_k_means:
	result = kmeans.predict(line)
	testes[result] += 1 

testes = testes * 100/num_testes
print(testes)


comb = combinations(np.arange(centers), 2)
dissi = np.zeros((centers, centers))

total = 0
for c1, c2 in comb:
	dissi[c2,c1] = np.sqrt(np.sum((kmeans.cluster_centers_[c1] - kmeans.cluster_centers_[c2])**2))
	if dissi[c2,c1] < 0.6:
		print(c1, c2)
		total+=1



saida = open("saida.txt", "w")
toSaida = "\t\t"
for f in file_var:
	for atr in atributos[f]:
		toSaida += atr + '\t'
toSaida += '\n Centroids gerados:\n'


toSaida += "\t\t"
for c in range(centers):
	for f in file_var:
		for value in kmeans.cluster_centers_[c]:
			toSaida += "{:.2f}".format(value) + "\t"
	toSaida += "\n\t\t"

for c in range(centers):
	for d in dissi[c][:c]:
		toSaida += "{:.4f}".format(d) + "\t"
	toSaida += "\n\t\t"
saida.write(toSaida)

print(total)

