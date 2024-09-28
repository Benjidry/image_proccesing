import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

# --- Funciones auxiliares ---
def create_spatial_affinity_kernel(spatial_sigma: float, size: int = 15):
    """Crear kernel gaussiano para la afinidad espacial."""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-0.5 * (dist ** 2) / (spatial_sigma ** 2))
    return kernel / np.sum(kernel)

def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    """Calcular los pesos de suavidad usando un kernel de afinidad espacial."""
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)

def refine_illumination_map(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Refinar el mapa de iluminación usando una matriz Laplaciana dispersa."""
    wx = compute_smoothness_weights(L, 1, kernel, eps)
    wy = compute_smoothness_weights(L, 0, kernel, eps)
    
    n, m = L.shape
    L_flatten = L.flatten()
    row, column, data = [], [], []

    # Construir la matriz dispersa
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            row.append(idx)
            column.append(idx)
            diag = 0

            if i > 0:  # Arriba
                idx_up = (i - 1) * m + j
                weight_up = wy[i - 1, j]
                row.append(idx)
                column.append(idx_up)
                data.append(-weight_up)
                diag += weight_up

            if i < n - 1:  # Abajo
                idx_down = (i + 1) * m + j
                weight_down = wy[i, j]
                row.append(idx)
                column.append(idx_down)
                data.append(-weight_down)
                diag += weight_down

            if j > 0:  # Izquierda
                idx_left = i * m + (j - 1)
                weight_left = wx[i, j - 1]
                row.append(idx)
                column.append(idx_left)
                data.append(-weight_left)
                diag += weight_left

            if j < m - 1:  # Derecha
                idx_right = i * m + (j + 1)
                weight_right = wx[i, j]
                row.append(idx)
                column.append(idx_right)
                data.append(-weight_right)
                diag += weight_right

            data.append(diag)
    
    A = csr_matrix((data, (row, column)), shape=(n * m, n * m))
    Id = diags([np.ones(n * m)], [0])
    L_refined_flat = spsolve(Id + lambda_ * A, L_flatten)
    L_refined = L_refined_flat.reshape((n, m))

    return np.clip(L_refined, eps, 1) ** gamma

def correct_underexposure(image: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Corregir subexposición en la imagen."""
    L = np.max(image, axis=-1)
    L_refined = refine_illumination_map(L, gamma, lambda_, kernel, eps)
    L_refined_3d = np.repeat(L_refined[:, :, np.newaxis], 3, axis=2)
    return image / L_refined_3d

def fuse_exposure(image, under_exposed, over_exposed, bc=1, bs=1, be=1):
    """Fusionar imágenes subexpuestas y sobreexpuestas usando Mertens fusion."""
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype(np.uint8) for x in [image, under_exposed, over_exposed]]
    fused_image = merge_mertens.process(images)
    return fused_image

def enhance_image_exposure(image: np.ndarray, gamma=2.2, lambda_=0.15, dual=True, sigma=3):
    """Mejorar la exposición de la imagen utilizando el método Dual."""
    kernel = create_spatial_affinity_kernel(sigma)
    image_normalized = image.astype(np.float32) / 255.0

    # Corregir subexposición
    under_corrected = correct_underexposure(image_normalized, gamma, lambda_, kernel)

    if dual:
        # Corregir sobreexposición (inverso de la imagen)
        inverted_image = 1 - image_normalized
        over_corrected = 1 - correct_underexposure(inverted_image, gamma, lambda_, kernel)
        # Fusionar ambas correcciones
        result = fuse_exposure(image_normalized, under_corrected, over_corrected)
    else:
        result = under_corrected

    return np.clip(result * 255, 0, 255).astype(np.uint8)

# --- Uso del método Dual Illumination Estimation ---
if __name__ == '__main__':
    # Cargar imagen de entrada
    image_path = 'tu_imagen.jpg'  # Reemplaza con la ruta de tu imagen
    image = cv2.imread(image_path)

    # Mejorar la exposición utilizando DUAL
    corrected_image = enhance_image_exposure(image, gamma=2.2, lambda_=0.15, dual=True, sigma=3)

    # Mostrar imagen original y mejorada
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen Mejorada', corrected_image)
    # Guardar imagen mejorada
    cv2.imwrite('imagen_mejorada.jpg', corrected_image)
