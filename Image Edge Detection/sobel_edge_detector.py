import cv2
import numpy as np
import matplotlib.pyplot as plt

images = ["license_plate.png", "lena.jpg"]
threshold = 100

def sobel_edge_detection(image_path, threshold):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 
    
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
    
    _, binary_edges = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    
    return img, gradient_magnitude, binary_edges

for image_path in images:
    original, gradient, edges = sobel_edge_detection(image_path, threshold)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Gradient Magnitude')
    plt.imshow(gradient, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.show()