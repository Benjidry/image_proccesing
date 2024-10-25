import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np


def equalize_histogram_hsv(img):
    # Convertir de BGR a HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Ecualizar el histograma solo en el canal V (Brillo)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    
    # Convertir de nuevo de HSV a BGR
    img_equalized = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # Retornar la imagen ecualizada
    return img_equalized

def equalize_histogram_srgb(img):
    # Convertir de sRGB a RGB lineal
    def srgb_to_linear(img):
        img = img / 255.0  # Normalizar a rango [0, 1]
        linear_mask = img <= 0.04045
        img[linear_mask] = img[linear_mask] / 12.92
        img[~linear_mask] = ((img[~linear_mask] + 0.055) / 1.055) ** 2.4
        return img

    # Convertir de RGB lineal a sRGB
    def linear_to_srgb(img):
        srgb_mask = img <= 0.0031308
        img[srgb_mask] = img[srgb_mask] * 12.92
        img[~srgb_mask] = 1.055 * (img[~srgb_mask] ** (1.0 / 2.4)) - 0.055
        img = np.clip(img * 255.0, 0, 255)  # Escalar de vuelta a [0, 255]
        return img.astype(np.uint8)

    # Convertir la imagen de sRGB a RGB lineal
    img_linear = srgb_to_linear(img.astype(np.float32))

    # Separar los canales de color R, G, B
    r, g, b = cv2.split(img_linear)

    # Aplicar ecualización de histograma en cada canal (convertido a 8 bits para ecualización)
    r_eq = cv2.equalizeHist((r * 255).astype(np.uint8))
    g_eq = cv2.equalizeHist((g * 255).astype(np.uint8))
    b_eq = cv2.equalizeHist((b * 255).astype(np.uint8))

    # Recombinar los canales después de ecualización (volver a rango [0, 1])
    img_linear_eq = cv2.merge([r_eq, g_eq, b_eq]).astype(np.float32) / 255.0

    # Convertir de nuevo de RGB lineal a sRGB
    img_srgb_eq = linear_to_srgb(img_linear_eq)

    return img_srgb_eq

def equalize_rgb(image):
    # Separar los canales de la imagen
    r, g, b = cv2.split(image)
    
    # Aplicar la ecualización de histograma a cada canal
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    
    # Combinar los canales ecualizados en una imagen
    image_eq = cv2.merge([r_eq, g_eq, b_eq])
    
    return image_eq

def gaussian_smoothing(img, kernel_size=(5, 5), sigma=0):
    # Aplicar el filtro Gaussiano
    img_smoothed = cv2.GaussianBlur(img, kernel_size, sigma)
    return img_smoothed

def equalize_histogram_hsv_clahe(img):
    # Convertir de BGR a HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Separar los canales H, S y V
    h, s, v = cv2.split(img_hsv)
    
    # Crear un objeto CLAHE (Clip Limit controla cuánto contraste se limita)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Aplicar CLAHE solo al canal V (Brillo)
    v_clahe = clahe.apply(v)
    
    # Recombinar los canales H, S y el nuevo V ajustado
    img_hsv_clahe = cv2.merge((h, s, v_clahe))
    
    # Convertir de nuevo de HSV a BGR
    img_clahe = cv2.cvtColor(img_hsv_clahe, cv2.COLOR_HSV2BGR)
    
    return img_clahe


class FruitApp(App):
    kv_file = None 

    def load_kv(self, *args, **kwargs):
        pass

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Widget de imagen
        self.image = Image()
        self.layout.add_widget(self.image)

        # Botón para cerrar
        self.button = Button(text="Cerrar", size_hint=(1, 0.1))
        self.button.bind(on_press=self.stop)
        self.layout.add_widget(self.button)

        # Iniciar la captura de video
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Actualizar 30 veces por segundo

        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            print(frame)
            #frame = equalize_rgb(frame)
            frame = gaussian_smoothing(frame, (9,9), 1)
            # Convertir la imagen para mostrarla en Kivy
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
        

    def on_stop(self):
        # Liberar la cámara cuando la aplicación se cierra
        self.capture.release()

# Ejecutar la aplicación
if __name__ == '__main__':
    FruitApp().run()

