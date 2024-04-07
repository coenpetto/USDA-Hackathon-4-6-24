import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

csv_path = 'Path1/Path1-Model Training/Path1 Challenge Training Data.csv'
image_folder_path = 'Path1/Path1-Model Training/Path1 Challenge Training Images'

df = pd.read_csv(csv_path)

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    return image

def prepare_dataset(df, image_folder_path):
    images = []
    labels = []
    
    label_mapping = {label: idx for idx, label in enumerate(df['Grade Category'].unique())}
    
    for _, row in df.iterrows():
        image_path = os.path.join(image_folder_path, row['Filename'])
        image = preprocess_image(image_path)
        label = label_mapping[row['Grade Category']]
        
        images.append(image)
        labels.append(label)
        
    return np.array(images), np.array(labels), label_mapping

def split_dataset(images, labels):
    return train_test_split(images, labels, test_size=0.2, random_state=42)

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    images, labels, label_mapping = prepare_dataset(df, image_folder_path)
    X_train, X_test, y_train, y_test = split_dataset(images, labels)
    model = create_model(X_train[0].shape, len(label_mapping))
    
    checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_accuracy')
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_cb]
    )
    
    print("Training complete. Best model saved as 'best_model.keras'.")
    
    val_accuracy = max(history.history['val_accuracy'])
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == '__main__':
    main()
