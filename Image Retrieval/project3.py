import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_corel5k_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
    return images

query_images = [
    cv2.imread(r"C:\Users\malih\Desktop\Project design in computer science\projects\project3\query images\query_images1"),
    cv2.imread(r"C:\Users\malih\Desktop\Project design in computer science\projects\project3\query images\query_images2"),
    cv2.imread(r"C:\Users\malih\Desktop\Project design in computer science\projects\project3\query images\query_images3")
]

corel_images = load_corel5k_images(r"C:\Users\malih\Desktop\Project design in computer science\Corel5k")

def extract_features(image):
    return np.random.rand(4096) 


query_features = [extract_features(img) for img in query_images]
corel_features = [extract_features(img) for img in corel_images]

results = []
N = 10 
for q_feat in query_features:
    similarities = cosine_similarity(q_feat.reshape(1, -1), corel_features)
    ranked_indices = np.argsort(similarities[0])[::-1] 
    results.append(ranked_indices[:N]) 

def display_results(query_images, corel_images, results):
    plt.figure(figsize=(15, 8))
    
    for i, result in enumerate(results):
        plt.subplot(len(query_images), N + 1, i * (N + 1) + 1)
        plt.imshow(cv2.cvtColor(query_images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Query Image {i + 1}')
        plt.axis('off')
        
        for j, index in enumerate(result):
            plt.subplot(len(query_images), N + 1, i * (N + 1) + j + 2)
            plt.imshow(cv2.cvtColor(corel_images[index], cv2.COLOR_BGR2RGB))
            plt.title(f'Match {j + 1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

display_results(query_images, corel_images, results)