
# Importando as bibliotecas necessárias

# Biblioteca OpenCV, que fornece as funções para manipulação e detecção de imagens
import cv
# Biblioteca numpy disponibiliza as funções utilizadas para eftuar alguns calculos
import numpy as np

# Carregando o algoritimo Yolo, são os arquivos nescessários para realizar detecção de objetos na imagem.

# Para a execução do algoritimo são necessarios tres arquivos:
# Weight file: arquivo com pesos de modeleo pré-treinado. É o núcleo do algoritimo para detectar os objetos na imagem.
# Cfg file: é o arquivo de configuração, onde existem todas as configurações do algoritimo.
# Nome files: contém o nome(Label) dos objetos que o algoritimo pode detectar. Lembrando que o YOLO consegue detectar
# até 80 classes de objetos diferentes em uma imagem ou em um frame de vídeo

''' 
Dessa forma usamos somente os pesos (weights) do modelo treinado para realizar as previsões.
Vale ressaltar que usar modelos -pré treinados acelara muito a implementação do aplicação, afinal,
não é necessario possuir um grande conjunto de imagens e nem gastar horas treinando o seu próprio modelo.
'''

net = cv.dnn.readNet('', '')
classes = []
# Obtém as classes dos objetos que o YOLO consegue detectar
with open('','r')as f:
    # Objeto que armazena as classes disponiveis, ou seja, os nomes dos objetos que o YOLO consegue detectar
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# Determina os nomes das camadas (output) que precisamos do YOLO
out_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Inicializa uma lista de cores para representar cada etiqueta de classes possível
colors = np.random.uniform(0, 255, size=(len(classes), 3))