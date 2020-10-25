
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
with open('Transito.jpg', 'r')as f:
    # Objeto que armazena as classes disponiveis, ou seja, os nomes dos objetos que o YOLO consegue detectar
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# Determina os nomes das camadas (output) que precisamos do YOLO
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Inicializa uma lista de cores para representar cada etiqueta de classes possível
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Carregando imagem. O objetivo é detectar objetos nesta imagem usando o YOLO.
img = cv.imread('')
# Realizando o redmensionamento da imagem.
img = cv.resize(img, None, fx=0.4, fy=0.4)
# Obetmos largura, altura e canais da imagem
height, width, channels = img.shape

# É necessário converter a imagem para o formato BLOB. O Blob é usado para extrair
# recursos de uma imagem e redmensioná-los.
# O YOLO aceita tres tamanhos:
# 320 x 320 é pequeno, tem menos precisão e mais velocidade
# 609 x 609 é grande, tem muita precisão e baixa velocidade.
# 416 x 416 é meio a meio, nem tão grande e nem tão rapido.

# Neste trecho convertermos a imagem e passamos ela para a rede neural fazer a detecção dos objetos.
# Além disto esta função agrupa a imagem de entrada, de modo que a rede neural possa fazer a inferencia em blocos.
# Como é possível verificar, foi definido que a imagem terá dimensão 416 x 416
blob = cv.dnn.blobFromImnage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Utilizamos a função net.forward() para realizar as inferencias ou predições sobre os grupos de imagens gerados,
# o resultado é um array com os valores de saída da rede.

# Outs é o array que contém o resultado da detecção.
# Ele fornece todas as informações sobre os objetos detectados pela rede, tal como sua posição e a confiança sobre a
# detecção.
outs = net.forward(output_layers)

# arrays que armazenarão as informações obtidas com resultado da detecção
class_ids = [] # Rotulo de classe do objeto detectado,
confidences = [] # Valor de confiança que o YOLO atribuiu ao objeto detectado.
boxes = [] # Caixas delimitadores ao redos do objeto (bound boxes)

# Estrutura de repetição que extrai as informações de detecção.
#  Ocorre sobre cada uma das saídas do comando
for ou in outs:
    #loop sobre cada uma das detecções
    for detection in outs:
        scores = detection[5:]
        class_id = np.argmaz(scores)
        confidences = scores[class_id]

        # Determinando o nível de confiança para detecção. Neste caso, maior que 0.5.
        # Definimos um limiar de confiança de 0.5, se for maior, consideramos o objeto corretamente detectado,
        # Caso contrário, ignoramos ele. O limiar vai de 0 a 1. Quanto mais proximo de 1 maior é a precisão da detecção,
        # enquanto que quanto mais próximo de 0 menor é a precisão, mas também é maior o número de objetos detectados.
        if confidences >0.5:
            # Coordenados da caixa delimitadora do objeto detectado para que possamos exibí-las na imagem original.

            # O algoritimo YOLO retornara as coordenadas do centro (x, y) da caixa delimitadora
            # bem como a largura e altura das caixas.
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordenadas do retangulo.
            # Usa o centro (x, y) para derivar o topo e conto esquerdo da caixa delimitadora.
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Atualiza a lista de coordenadas da caixa delimitadora, confiaça da detecção, e IDs de classe.
            boxes.append([x, y, w, h])
            confidences.append(float(confidences))
            class_ids.append(class_id)

            color = colors[0]

            # Retangulo representando as caixas delimitadoras dos objetos detectados.
            cv.rectangle(img, (x,y), (x + w, y), color, 2)

            # Desenha um circulo verde que identifica o centro da imagem do objeto detectados.
            cv.circle(img, (center_y, center_y), 10, (0, 255, 0), 2)


# Quando realizamos a detecção, pode acontecer de existir mais de uma caixa deleimitadora para o mesmo objeto,
# então devemos usar outra função de remover esse "ruido". Isto é chamado de supressão não máxima.
# Está função suprime caixas delimitadoras significantes sobrepostas, mantendo apenas as mais confiantes.
# Esse algoritimo foi aplicado com o módulo DNN na função cv.dnn.NMSBoxes().
indexes = cv.dnn.MSBoxes(boxes, confidences, 0.5, 0.4)
# print(indexes)

# Determine qual será a fonte do texto que irá exibir as informações dos objetos detectados.
# Neste caso será a fonte sans-serif de tamanho pequeno.
font = cv.FONT_HERSHEY_PLAIN

# Finalmente extraímos todas as informações e as mostramos na tela.
# Box: contém as coordenadas do retangulo ao redos do objeto detectado
# Label: é o nome do objeto detectado.
# Confidence: a confiança sobre a detecção de 0 a 1.

# Estrutura de repetição que define as informações que serão exibidas na tela.
# Tais como a imagem original, as caixas delimitado
for i in range(len(boxes)):
    # Loop sobre os indeices
    if i in indexes:
        # Obtém as coordenadas que fomarão o retangulo que indentifica a localizção do obtejo detectado
        x, y, w, h = boxes[i]
        # Obtém o rotolo do objeto detectado.
        label = str(classes[class_ids[i]])
        # Obtém o valor de nível de confiança da detecção
        confidence = confidences[i]
        color = colors[i]
        # Desenha o retangulo que representa a caiza delimitadora e define a cor dele.
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # Apresenta o nome do objeto detectado(Label) junto ao nível de confiança (confidence) sobre a detecção.
        cv.putText(img, label + "" + str(round(confidence)), (x, y + 30), font, 3, color, 3)