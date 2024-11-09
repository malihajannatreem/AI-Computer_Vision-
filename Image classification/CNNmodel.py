import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PIL import Image
import os

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

cnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

cnn_predictions = np.argmax(cnn_model.predict(val_images), axis=1)
cnn_accuracy = accuracy_score(val_labels, cnn_predictions)
cnn_precision = precision_score(val_labels, cnn_predictions, average='weighted')
cnn_recall = recall_score(val_labels, cnn_predictions, average='weighted')

print(f"CNN Accuracy: {cnn_accuracy}, Precision: {cnn_precision}, Recall: {cnn_recall}")