import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Paths
csv_path = 'Path1/Path1-Model Training/Path1 Challenge Training Data.csv'
image_folder_path = 'Path1/Path1-Model Training/Path1 Challenge Training Images'
model_path = '9537.keras'  # Adjust the model filename if necessary

# Function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    return image

# Prepare the dataset
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

# Load and preprocess the dataset
df = pd.read_csv(csv_path)
images, labels, label_mapping = prepare_dataset(df, image_folder_path)
_, X_test, _, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the model
model = keras.models.load_model(model_path)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, labels, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

# Predict
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
