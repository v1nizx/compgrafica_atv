from cv2.gapi import crop
!pip install opencv-python matplotlib numpy
import cv2
import numpy as np
import matplotlib.pyplot as plt


#carregar imagens em escala de cinza

img = cv2.imread('slide17.png', cv2.IMREAD_GRAYSCALE)

#aplicar dft

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

#calcular magnitude

magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_log = np.log(magnitude+1)

plt.subplot(1,2,1)
plt.imshow(magnitude_log, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')
plt.title('Espectro de Fourier')
plt.show()


dft_noshift = np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1])+1)
plt.subplot(1,2,1)
plt.imshow(dft_noshift, cmap='gray')
plt.title('Espectro sem deslocamento')

plt.subplot(1,2,2)
plt.imshow(magnitude_log, cmap='gray')
plt.title('Espectro com deslocamento')
plt.show()


rows, cols = img.shape
crow, ccol = rows//2, cols//2

mask = np.ones((rows, cols, 2), np.float32)
r = 30
center = [crow, ccol]
cv2.circle(mask, (ccol,crow), r, (0,0,0), -1)

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.imshow(img_back, cmap='gray')
plt.title('Filtro Passa-Alta (Laplaciano)')
plt.show()

x = np.linspace(-ccol, ccol, cols)
y = np.linspace(-crow, crow, rows)
x, y = np.meshgrid(x, y)
sigma = 30
gaussian = np.exp(-(x**2+y**2)/(2*sigma**2))
gaussian = gaussian/gaussian.max()

mask_gauss = np.zeros((rows, cols, 2), np.float32)
mask_gauss[:, :, 0] = gaussian
mask_gauss[:, :, 1] = gaussian

fshift_gauss = dft_shift*mask_gauss
f_ishift = np.fft.ifftshift(fshift_gauss)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.imshow(img_back, cmap='gray')
plt.title('Filto Passa-Baixa (Gaussiano)')
plt.show()

img_float = np.float32(img)/255.0
dct = cv2.dct(img_float)

dct_compress = dct.copy()
dct_compress[50:,:] = 0
dct_compress[:,50:] = 0

img_recon = cv2.idct(dct_compress)

plt.subplot(1,2,1)
plt.imshow(img_float, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1,2,2)
plt.imshow(img_recon, cmap='gray')
plt.title('Imagem Reconstruida (DCT)')
plt.show()
