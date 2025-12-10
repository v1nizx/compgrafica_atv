from cv2.gapi import threshold
!pip install opencv-python matplotlib numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('slide17.png', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Detecção de Bordas (Canny)')
plt.show()

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

kernel_roberts_x = np.array([[1, 0], [0, -1]])
kernel_roberts_y = np.array([[0, 1], [-1, 0]])
robertsx = cv2.filter2D(img, -1, kernel_roberts_x)
robertsy = cv2.filter2D(img, -1, kernel_roberts_y)
roberts = cv2.magnitude(robertsx.astype(float), robertsy.astype(float))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Filtro Sobel')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Filtro Roberts')
plt.show()

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)
eroded = cv2.erode(img, kernel, iterations=1)
gradient = dilated - eroded

plt.imshow(gradient, cmap='gray')
plt.title('Gradiente Morfológico')
plt.show

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1,2,2)
plt.imshow(thresh, cmap='gray')
plt.title('Segmentação por Limiar')
plt.show()

img_color = cv2.imread('slide36.png')
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
sobel = cv2.convertScaleAbs(sobel)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, markers = cv2.threshold(dist, 0.7*dist.max(), 255, 0)
markers = np.uint8(markers)

markers = cv2.connectedComponents(markers)[1]
markers = cv2.watershed(img_color, markers)
img_color[markers == -1] = [255, 0, 0]

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Segmentação por Watershed')
plt.show()
