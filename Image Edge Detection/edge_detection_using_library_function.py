import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

lena = io.imread('lena.jpg')

gray_lena = color.rgb2gray(lena)

# Sobel edge detection
sobel_edges = cv2.Sobel(np.float32(gray_lena), cv2.CV_64F, 1, 1, ksize=3)

# Prewitt edge detection (using a custom kernel since OpenCV doesn't have it natively)
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_edges_x = cv2.filter2D(gray_lena, -1, prewitt_kernel_x)
prewitt_edges_y = cv2.filter2D(gray_lena, -1, prewitt_kernel_y)
prewitt_edges = np.sqrt(prewitt_edges_x**2 + prewitt_edges_y**2)

# Roberts edge detection
roberts_kernel_x = np.array([[1, 0], [0, -1]])
roberts_kernel_y = np.array([[0, 1], [-1, 0]])
roberts_edges_x = cv2.filter2D(gray_lena, -1, roberts_kernel_x)
roberts_edges_y = cv2.filter2D(gray_lena, -1, roberts_kernel_y)
roberts_edges = np.sqrt(roberts_edges_x**2 + roberts_edges_y**2)

# Canny edge detection
canny_edges = cv2.Canny(np.uint8(gray_lena*255), 100, 200)

# Display results
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel Edge Detection')

plt.subplot(2, 2, 2)
plt.imshow(prewitt_edges, cmap='gray')
plt.title('Prewitt Edge Detection')

plt.subplot(2, 2, 3)
plt.imshow(roberts_edges, cmap='gray')
plt.title('Roberts Edge Detection')

plt.subplot(2, 2, 4)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.show()