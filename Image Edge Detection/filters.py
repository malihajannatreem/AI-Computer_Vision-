import cv2
import numpy as np
import matplotlib.pyplot as plt

image_license_plate = cv2.imread('license_plate.png', cv2.IMREAD_GRAYSCALE)
image_lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

sobel_horizontal = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

sobel_vertical = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]])

def apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

horizontal_license_plate = apply_filter(image_license_plate, sobel_horizontal)
vertical_license_plate = apply_filter(image_license_plate, sobel_vertical)
horizontal_lena = apply_filter(image_lena, sobel_horizontal)
vertical_lena = apply_filter(image_lena, sobel_vertical)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(image_license_plate, cmap='gray')
axes[0, 0].set_title('License Plate (Original)')
axes[0, 1].imshow(horizontal_license_plate, cmap='gray')
axes[0, 1].set_title('License Plate (Horizontal Edges)')
axes[0, 2].imshow(vertical_license_plate, cmap='gray')
axes[0, 2].set_title('License Plate (Vertical Edges)')

axes[1, 0].imshow(image_lena, cmap='gray')
axes[1, 0].set_title('Lena (Original)')
axes[1, 1].imshow(horizontal_lena, cmap='gray')
axes[1, 1].set_title('Lena (Horizontal Edges)')
axes[1, 2].imshow(vertical_lena, cmap='gray')
axes[1, 2].set_title('Lena (Vertical Edges)')

plt.show()