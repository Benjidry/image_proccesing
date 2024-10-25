import cv2
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('runs/detect/train/weights/last.pt')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Realizar la predicción
    results = model(frame)

    # Dibujar las cajas delimitadoras y etiquetas en el frame
    annotated_frame = results[0].plot()  # Usamos .plot() para visualizar

    # Mostrar el frame con las detecciones
    cv2.imshow('Detección de cara en tiempo real', annotated_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()