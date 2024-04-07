import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from PIL import Image
from tensorflow.keras.optimizers import Adam
from collections import defaultdict

csv_path = 'Path1/Path1-Model Training/Path1 Challenge Training Data.csv'
image_folder_path = 'Path1/Path1-Model Training/Path1 Challenge Training Images'
model_path = 'best_model.keras'

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    return image

def score_to_grade(score):
    if 0 <= score < 400:
        return 'Select'
    elif 400 <= score < 499:
        return 'Low Choice'
    elif 500 <= score < 699:
        return 'Upper 2/3 Choice'
    elif 700 <= score <= 1100:
        return 'Prime'
    else:
        return 'Unknown'

def prepare_dataset(df, image_folder_path):
    images = []
    scores = []
    
    for _, row in df.iterrows():
        image_path = os.path.join(image_folder_path, row['Filename'])
        image = preprocess_image(image_path)
        score = row['Score']
        
        images.append(image)
        scores.append(score)
        
    return np.array(images), np.array(scores)

def evaluate(y_true, y_pred):
    correct = 0
    error_count = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
        else:
            error_count[true] += 1
            print(f"Correct: {true}, Predicted: {pred}")
    
    accuracy = correct / len(y_true)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    if error_count:
        print("\nError Summary by True Label:")
        for label, count in error_count.items():
            print(f"Label {label}: {count} errors")
    
    return accuracy

df = pd.read_csv(csv_path)
images, scores = prepare_dataset(df, image_folder_path)
X_train, X_test, y_train, y_test_scores = train_test_split(images, scores, test_size=0.2, random_state=42)

y_test_grades = [score_to_grade(score) for score in y_test_scores]

model = keras.models.load_model(model_path)

# run inferences on the test set
predicted_scores = model.predict(X_test).flatten()
predicted_grades = [score_to_grade(score) for score in predicted_scores]

for i in range(20):
    print(f"Actual: {y_test_grades[i]}, Predicted: {predicted_grades[i]}")

# calculate the accuracy of grade classification
# accuracy = accuracy_score(y_test_grades, predicted_grades)

accuracy = evaluate(y_test_grades, predicted_grades)
print(min(predicted_scores), max(predicted_scores))
print(f"Grade classification accuracy: {accuracy:.4f}")
