import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from dual_ilumin import enhance_image_exposure
import cv2
import numpy as np


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

def equalize_rgb_lab(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# Separar los canales de LAB (L, A, B)
    l, a, b = cv2.split(lab_img)
# Ecualizar el histograma solo en el canal L (luminosidad)
    l_eq = cv2.equalizeHist(l)
# Unir los canales de nuevo (L ecualizado, A y B originales)
    lab_eq = cv2.merge([l_eq, a, b])
# Convertir la imagen de nuevo a RGB
    img_rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return img_rgb_eq

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
            frame = enhance_image_exposure(frame, gamma=2.2, lambda_=0.15, dual=True, sigma=3)

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

