import cv2
import numpy as np
from dual_ilumin import enhance_image_exposure

def real_time_dual_illumination():
    # Inicializar la captura de video de la webcam (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)

    # Comprobar si la webcam está abierta
    if not cap.isOpened():
        print("Error: No se puede acceder a la webcam")
        return

    # Ajustar parámetros del algoritmo Dual Illumination
    gamma = 2.2
    lambda_ = 0.15
    dual = True
    sigma = 3

    # Procesar cada fotograma de la webcam en tiempo real
    while True:
        # Leer un fotograma de la webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede leer el fotograma de la webcam")
            break

        # Convertir la imagen a float32 y normalizar (0-1) para el procesamiento
        frame_normalized = frame.astype(np.float32) / 255.0
        
        # Aplicar la mejora de exposición usando Dual Illumination
        enhanced_frame = enhance_image_exposure(frame_normalized, gamma, lambda_, dual, sigma)
        
        # Mostrar el fotograma procesado
        cv2.imshow('Dual Illumination Webcam', enhanced_frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

real_time_dual_illumination()
