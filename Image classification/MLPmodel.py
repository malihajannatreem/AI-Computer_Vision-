import os
import numpy as np
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split  
from PIL import Image  

def load_dataset(data_directory):
    images = []
    labels = []
    class_names = os.listdir(data_directory)
    
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_directory, class_name)
        print(f"Checking directory: {class_dir}") 

        if not os.path.isdir(class_dir):
            print(f"Skipping {class_dir} because it's not a directory.")
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).resize((128, 128)) 
                img_array = np.array(img) / 255.0 
                images.append(img_array)
                labels.append(class_index)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), len(class_names)

data_directory = r'C:\Users\malih\Desktop\Project4New\Corel5k'
images, labels, num_classes = load_dataset(data_directory)

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

train_images_flat = train_images.reshape(-1, 128 * 128 * 3)
val_images_flat = val_images.reshape(-1, 128 * 128 * 3)

mlp_model = models.Sequential([
    layers.Input(shape=(128 * 128 * 3,)),  
    layers.Dense(256, activation='relu'),  
    layers.Dense(128, activation='relu'),  
    layers.Dense(num_classes, activation='softmax')  
])

mlp_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

mlp_model.fit(train_images_flat, train_labels, epochs=10, validation_data=(val_images_flat, val_labels))

mlp_predictions = np.argmax(mlp_model.predict(val_images_flat), axis=1)
mlp_accuracy = accuracy_score(val_labels, mlp_predictions)
mlp_precision = precision_score(val_labels, mlp_predictions, average='weighted')
mlp_recall = recall_score(val_labels, mlp_predictions, average='weighted')
print(f'Accuracy: {mlp_accuracy}')
print(f'Precision: {mlp_precision}')
print(f'Recall: {mlp_recall}')