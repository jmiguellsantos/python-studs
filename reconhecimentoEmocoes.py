import cv2
import mediapipe as mp
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model
import numpy as np

# Configurações para o reconhecimento facial
reconhecimento_rosto = mp.solutions.face_detection
desenho = mp.solutions.drawing_utils
reconhecedor_rosto = reconhecimento_rosto.FaceDetection(min_detection_confidence=0.2)

# Carregar o modelo de emoções (substitua pelo seu modelo treinado)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(7, activation='softmax')(x)
modelo_emocoes = Model(inputs=base_model.input, outputs=x)

# Mapear as emoções detectadas
emocao_map = {
    0: "Raiva",
    1: "Desprezo",
    2: "Medo",
    3: "Feliz",
    4: "Triste",
    5: "Surpreso",
    6: "Neutro"
}

# Função para identificar a emoção
def identificar_emocao(imagem):
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    imagem_cropped = cv2.resize(imagem_rgb, (48, 48))
    imagem_cropped = np.expand_dims(imagem_cropped, axis=0)
    imagem_cropped = preprocess_input(imagem_cropped)
    predicoes = modelo_emocoes.predict(imagem_cropped)
    emocao = np.argmax(predicoes[0])
    return emocao_map[emocao]

# Função para mostrar a imagem da webcam e identificar emoções
def mostrar_imagem():
    webcam = cv2.VideoCapture(0)
    while webcam.isOpened():
        validacao, frame = webcam.read()  # Ler a imagem da webcam
        if not validacao:
            break
        imagem = frame
        lista_rostos = reconhecedor_rosto.process(imagem)  # Processar a imagem para reconhecer rostos
        
        num_rostos = 0  # Contador de rostos detectados
        if lista_rostos.detections:  # Se algum rosto for reconhecido
            num_rostos = len(lista_rostos.detections)
            for rosto in lista_rostos.detections:  # Para cada rosto reconhecido
                desenho.draw_detection(imagem, rosto)  # Desenhar a caixa ao redor do rosto

                # Coordenadas do rosto detectado
                x_min = int(rosto.location_data.relative_bounding_box.xmin * imagem.shape[1])
                y_min = int(rosto.location_data.relative_bounding_box.ymin * imagem.shape[0])
                x_max = int((rosto.location_data.relative_bounding_box.xmin + rosto.location_data.relative_bounding_box.width) * imagem.shape[1])
                y_max = int((rosto.location_data.relative_bounding_box.ymin + rosto.location_data.relative_bounding_box.height) * imagem.shape[0])
                
                rosto_imagem = imagem[y_min:y_max, x_min:x_max]
                emocao = identificar_emocao(rosto_imagem)
                
                # Exibir a emoção na imagem
                cv2.putText(imagem, emocao, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Mensagens específicas
                if emocao == "Feliz":
                    cv2.putText(imagem, "Você está Feliz", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif emocao == "Desprezo":
                    cv2.putText(imagem, "Legal", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif emocao == "Raiva":
                    cv2.putText(imagem, "Tchau", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Exibir o número de pessoas detectadas
        cv2.putText(imagem, f"Número de Pessoas: {num_rostos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar a imagem da webcam
        cv2.imshow("Detecção de Emoções", imagem)
        if cv2.waitKey(5) == 27:  # ESC para sair
            break

    webcam.release()  # Encerrar a conexão com a webcam
    cv2.destroyAllWindows()  # Fechar a janela da webcam

# Iniciar a captura de vídeo
mostrar_imagem()
