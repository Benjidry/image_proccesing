import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

def create_spatial_affinity_kernel(spatial_sigma: float, size: int = 15):
    """
    Crear un kernel gaussiano para calcular la afinidad espacial entre píxeles.
    
    Parámetros:
    - spatial_sigma (float): Controla la extensión de la suavidad en el kernel. A mayor valor, mayor suavidad.
    - size (int, opcional): El tamaño del kernel (matriz cuadrada de tamaño `size x size`). Por defecto es 15.

    Retorno:
    - kernel (np.ndarray): El kernel gaussiano normalizado que puede ser usado para suavizar imágenes.
    """
    # Inicializar un kernel vacío de tamaño `size x size`.
    kernel = np.zeros((size, size))
    
    # Calcular la posición central del kernel.
    center = size // 2
    
    # Rellenar el kernel con valores gaussianos basados en la distancia al centro.
    for i in range(size):
        for j in range(size):
            # Calcular la distancia euclidiana entre el punto (i, j) y el centro del kernel.
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            # Aplicar la función gaussiana a la distancia.
            kernel[i, j] = np.exp(-0.5 * (dist ** 2) / (spatial_sigma ** 2))
    
    # Normalizar el kernel dividiendo por la suma de todos los elementos.
    return kernel / np.sum(kernel)


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    """
    Calcular los pesos de suavidad para cada píxel basado en su variación de iluminación.
    
    Parámetros:
    - L (np.ndarray): Mapa de iluminación de la imagen.
    - x (int): Dirección de cálculo del gradiente (1 para horizontal, 0 para vertical).
    - kernel (np.ndarray): Kernel de afinidad espacial.
    - eps (float, opcional): Término de regularización para evitar divisiones por cero. Por defecto es 1e-3.

    Retorno:
    - T (np.ndarray): Matriz de pesos de suavidad.
    """
    # Calcular el gradiente de la imagen L en la dirección horizontal (x=1) o vertical (x=0) usando Sobel.
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    
    # Aplicar el kernel gaussiano a una matriz de unos (suavizar la matriz de unos del tamaño de L).
    T = convolve(np.ones_like(L), kernel, mode='constant')
    
    # Suavizar el gradiente de la imagen Lp usando el kernel y regularizar con `eps` para evitar división por cero.
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    
    # Dividir por el valor absoluto del gradiente de la imagen con `eps` para evitar divisiones por cero.
    return T / (np.abs(Lp) + eps)


def refine_illumination_map(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """
    Refinar el mapa de iluminación usando una matriz Laplaciana dispersa para suavizar las variaciones de iluminación.
    
    Parámetros:
    - L (np.ndarray): El mapa de iluminación original.
    - gamma (float): Parámetro de corrección de iluminación.
    - lambda_ (float): Factor de ponderación para la matriz de suavidad.
    - kernel (np.ndarray): Kernel de afinidad espacial.
    - eps (float, opcional): Término de regularización. Por defecto es 1e-3.

    Retorno:
    - L_refined (np.ndarray): Mapa de iluminación refinado.
    """
    # Calcular los pesos de suavidad para las direcciones horizontal (wx) y vertical (wy).
    wx = compute_smoothness_weights(L, 1, kernel, eps)
    wy = compute_smoothness_weights(L, 0, kernel, eps)
    
    # Obtener el tamaño de la imagen (número de filas y columnas).
    n, m = L.shape
    
    # Aplanar la imagen L para usarla en cálculos de matrices dispersas.
    L_flatten = L.flatten()
    
    # Inicializar listas para construir la matriz dispersa.
    row, column, data = [], [], []

    # Construir la matriz Laplaciana dispersa a partir de los pesos de suavidad.
    for i in range(n):
        for j in range(m):
            idx = i * m + j  # Índice lineal del píxel en la imagen aplanada.
            row.append(idx)
            column.append(idx)
            diag = 0  # Valor diagonal de la matriz.

            # Agregar conexiones con los vecinos (arriba, abajo, izquierda, derecha) en la matriz dispersa.
            if i > 0:  # Píxel superior.
                idx_up = (i - 1) * m + j
                weight_up = wy[i - 1, j]
                row.append(idx)
                column.append(idx_up)
                data.append(-weight_up)  # Contribución del vecino.
                diag += weight_up  # Aumentar el valor diagonal.

            if i < n - 1:  # Píxel inferior.
                idx_down = (i + 1) * m + j
                weight_down = wy[i, j]
                row.append(idx)
                column.append(idx_down)
                data.append(-weight_down)
                diag += weight_down

            if j > 0:  # Píxel izquierdo.
                idx_left = i * m + (j - 1)
                weight_left = wx[i, j - 1]
                row.append(idx)
                column.append(idx_left)
                data.append(-weight_left)
                diag += weight_left

            if j < m - 1:  # Píxel derecho.
                idx_right = i * m + (j + 1)
                weight_right = wx[i, j]
                row.append(idx)
                column.append(idx_right)
                data.append(-weight_right)
                diag += weight_right

            # Agregar el valor diagonal.
            data.append(diag)
    
    # Crear la matriz dispersa A usando las listas `row`, `column` y `data`.
    A = csr_matrix((data, (row, column)), shape=(n * m, n * m))
    
    # Crear la matriz identidad.
    Id = diags([np.ones(n * m)], [0])
    
    # Resolver el sistema lineal (Id + λ * A) * L_refined_flat = L_flatten.
    L_refined_flat = spsolve(Id + lambda_ * A, L_flatten)
    
    # Reestructurar L_refined_flat a su forma original (n x m).
    L_refined = L_refined_flat.reshape((n, m))

    # Retornar el mapa de iluminación refinado, ajustado con gamma y limitado entre eps y 1.
    return np.clip(L_refined, eps, 1) ** gamma

def correct_underexposure(image: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """
    Corregir problemas de subexposición en una imagen utilizando el refinamiento del mapa de iluminación.
    
    Parámetros:
    - image (np.ndarray): Imagen original en formato RGB o similar.
    - gamma (float): Parámetro para ajustar la corrección de iluminación.
    - lambda_ (float): Factor de ponderación para suavizar el mapa de iluminación.
    - kernel (np.ndarray): Kernel gaussiano utilizado para la afinidad espacial.
    - eps (float, opcional): Término de regularización para evitar divisiones por cero. Por defecto es 1e-3.

    Retorno:
    - image_corrected (np.ndarray): Imagen corregida en cuanto a subexposición.
    """
    # Extraer el canal máximo de cada píxel (máxima iluminación).
    L = np.max(image, axis=-1)
    
    # Refinar el mapa de iluminación usando la función `refine_illumination_map`.
    L_refined = refine_illumination_map(L, gamma, lambda_, kernel, eps)
    
    # Expandir el mapa de iluminación refinado a tres canales (RGB) para que coincida con la imagen original.
    L_refined_3d = np.repeat(L_refined[:, :, np.newaxis], 3, axis=2)
    
    # Dividir la imagen original por el mapa refinado para corregir la subexposición.
    return image / L_refined_3d


def fuse_exposure(image, under_exposed, over_exposed, bc=1, bs=1, be=1):
    """
    Fusionar imágenes subexpuestas, sobreexpuestas y originales usando la fusión de Mertens.
    
    Parámetros:
    - image (np.ndarray): Imagen original normalizada.
    - under_exposed (np.ndarray): Imagen corregida por subexposición.
    - over_exposed (np.ndarray): Imagen corregida por sobreexposición.
    - bc (float, opcional): Parámetro de contraste. Por defecto es 1.
    - bs (float, opcional): Parámetro de saturación. Por defecto es 1.
    - be (float, opcional): Parámetro de exposición. Por defecto es 1.

    Retorno:
    - fused_image (np.ndarray): Imagen fusionada.
    """
    # Crear el objeto de fusión de Mertens con los parámetros dados.
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    
    # Convertir las imágenes de float32 (0-1) a uint8 (0-255) para la fusión.
    images = [np.clip(x * 255, 0, 255).astype(np.uint8) for x in [image, under_exposed, over_exposed]]
    
    # Aplicar el proceso de fusión a las imágenes.
    fused_image = merge_mertens.process(images)
    
    # Retornar la imagen fusionada.
    return fused_image


def enhance_image_exposure(image: np.ndarray, gamma=2.2, lambda_=0.15, dual=True, sigma=3):
    """
    Mejorar la exposición de la imagen aplicando corrección de subexposición y sobreexposición.
    
    Parámetros:
    - image (np.ndarray): Imagen original en formato RGB o similar.
    - gamma (float, opcional): Parámetro de corrección de iluminación. Por defecto es 2.2.
    - lambda_ (float, opcional): Factor de suavizado del mapa de iluminación. Por defecto es 0.15.
    - dual (bool, opcional): Si es True, también se corregirá la sobreexposición. Por defecto es True.
    - sigma (float, opcional): Parámetro de suavidad para el kernel gaussiano. Por defecto es 3.

    Retorno:
    - result (np.ndarray): Imagen mejorada con la exposición corregida.
    """
    # Crear el kernel de afinidad espacial.
    kernel = create_spatial_affinity_kernel(sigma)
    
    # Normalizar la imagen a valores entre 0 y 1.
    image_normalized = image.astype(np.float32) / 255.0

    # Corregir subexposición de la imagen.
    under_corrected = correct_underexposure(image_normalized, gamma, lambda_, kernel)

    if dual:
        # Invertir la imagen para corregir la sobreexposición.
        inverted_image = 1 - image_normalized
        
        # Corregir la sobreexposición de la imagen invertida.
        over_corrected = 1 - correct_underexposure(inverted_image, gamma, lambda_, kernel)
        
        # Fusionar la imagen original, subexpuesta y sobreexpuesta.
        result = fuse_exposure(image_normalized, under_corrected, over_corrected)
    else:
        # Solo aplicar la corrección de subexposición si `dual` es False.
        result = under_corrected

    # Escalar la imagen resultante a valores de 0 a 255 y convertir a uint8.
    return np.clip(result * 255, 0, 255).astype(np.uint8)


# --- Uso del método Dual Illumination Estimation ---
if __name__ == '__main__':
    # Cargar imagen de entrada
    image_path = 'lowexp.jpg'  # Reemplaza con la ruta de tu imagen
    image = cv2.imread(image_path)

    # Mejorar la exposición utilizando DUAL
    corrected_image = enhance_image_exposure(image, gamma=2.2, lambda_=0.15, dual=True, sigma=3)

    # Mostrar imagen original y mejorada
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen Mejorada', corrected_image)
    # Guardar imagen mejorada
    cv2.imwrite('imagen_mejorada.jpg', corrected_image)
