import cv2
import mediapipe as mp
import numpy as np

# Configurações para o reconhecimento facial e de mãos
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar o detector de rosto e mãos
reconhecedor_rosto = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
reconhecedor_mao = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Função para identificar o gesto da mão
def identificar_gesto_mao(landmarks):
    # Aqui você pode implementar a lógica para identificar os gestos
    # Vamos usar uma lógica básica para simplificar
    if landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y:
        return "Legal"
    elif landmarks[0].y < landmarks[1].y and landmarks[0].y < landmarks[2].y:
        return "Tchau"
    return "Nenhum gesto reconhecido"

def mostrar_imagem():
    webcam = cv2.VideoCapture(0)
    while webcam.isOpened():
        validacao, frame = webcam.read()  # Ler a imagem da webcam
        if not validacao:
            break
        imagem = frame

        # Processar rosto
        lista_rostos = reconhecedor_rosto.process(imagem)
        if lista_rostos.detections:
            for rosto in lista_rostos.detections:
                mp_drawing.draw_detection(imagem, rosto)

        # Processar mãos
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        resultados_mao = reconhecedor_mao.process(imagem_rgb)
        if resultados_mao.multi_hand_landmarks:
            for mao in resultados_mao.multi_hand_landmarks:
                mp_drawing.draw_landmarks(imagem, mao, mp_hands.HAND_CONNECTIONS)

                # Identificar gesto da mão
                gesto = identificar_gesto_mao(mao.landmark)
                if gesto == "Legal":
                    cv2.putText(imagem, "Você fez um Legal!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif gesto == "Tchau":
                    cv2.putText(imagem, "Você fez um Tchau!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar a imagem da webcam
        cv2.imshow("Detecção de Gestos e Emoções", imagem)
        if cv2.waitKey(5) == 27:  # ESC para sair
            break

    webcam.release()  # Encerrar a conexão com a webcam
    cv2.destroyAllWindows()  # Fechar a janela da webcam

# Iniciar a captura de vídeo
mostrar_imagem()
