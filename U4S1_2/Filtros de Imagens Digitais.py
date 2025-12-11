import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data # Biblioteca que já tem imagens de exemplo

# 1. CARREGAR A IMAGEM DIRETO DA BIBLIOTECA (Sem download)
# A função data.camera() já retorna a imagem do Cameraman em preto e branco
img = data.camera()

# Configurar tamanho da plotagem
plt.figure(figsize=(12, 12))

# --- ORIGINAL ---
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original (Cameraman)")
plt.axis("off")

# --- Q1: NEGATIVO ---
negativo = 255 - img
plt.subplot(3, 3, 2)
plt.imshow(negativo, cmap='gray')
plt.title("Filtro Negativo")
plt.axis("off")

# --- Q2: LIMIARIZAÇÃO (THRESHOLD) ---
# Limiar 127: o que for menor vira preto, maior vira branco
_, limiar = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.subplot(3, 3, 3)
plt.imshow(limiar, cmap='gray')
plt.title("Limiarização (Binário)")
plt.axis("off")

# --- Q3: LAPLACIANO (Detecta bordas gerais) ---
laplaciano = cv2.Laplacian(img, cv2.CV_64F)
plt.subplot(3, 3, 4)
plt.imshow(np.abs(laplaciano), cmap='gray')
plt.title("Laplaciano")
plt.axis("off")

# --- Q4: SOBEL X (Bordas Verticais) ---
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
plt.subplot(3, 3, 5)
plt.imshow(np.abs(sobelx), cmap='gray')
plt.title("Sobel X (Verticais)")
plt.axis("off")

# --- Q4: SOBEL Y (Bordas Horizontais) ---
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
plt.subplot(3, 3, 6)
plt.imshow(np.abs(sobely), cmap='gray')
plt.title("Sobel Y (Horizontais)")
plt.axis("off")

# --- Q5: HISTOGRAMA ---
plt.subplot(3, 3, 7)
plt.hist(img.ravel(), 256, [0, 256], color='black')
plt.title("Histograma")

# --- Q6: FILTRO GAUSSIANO (Suavização) ---
suave = cv2.GaussianBlur(img, (5, 5), 0)
plt.subplot(3, 3, 8)
plt.imshow(suave, cmap='gray')
plt.title("Gaussiano (Suavização)")
plt.axis("off")

plt.tight_layout()
plt.show()
