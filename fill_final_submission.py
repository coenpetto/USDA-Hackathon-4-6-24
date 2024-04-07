import os
from PIL import Image
import numpy as np
from tensorflow import keras
import pandas as pd

answer_sheet_path = 'final_submission.xlsx'
image_folder_path = '/Users/wyatt/Documents/usdabeef/Path1/Path1-Model Validation/Path1 Challenge Images for Validation'
model_path = 'best_model3.keras'

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    return image

def fill_final_submission(answer_sheet_path, image_folder_path, model_path):
    model = keras.models.load_model(model_path)
    df = pd.read_excel(answer_sheet_path)
    
    files = sorted([f for f in os.listdir(image_folder_path) if f != '.DS_Store'])
    
    print(f"Files: {files}")

    for index, filename in enumerate(files):
        image_path = os.path.join(image_folder_path, filename)
        print(f"Processing image: {image_path}")
        image = preprocess_image(image_path)
        prediction = model.predict(np.array([image]))
        predicted_grade = np.argmax(prediction)
        print(f"Predicted grade: {predicted_grade}")

        df.at[index, 'Grade Category'] = predicted_grade
        
        if predicted_grade == 0:
            df.at[index, 'Grade Category'] = 'Select'
        elif predicted_grade == 1:
            df.at[index, 'Grade Category'] = 'Low Choice'
        elif predicted_grade == 2:
            df.at[index, 'Grade Category'] = 'Upper 2/3 Choice'
        elif predicted_grade == 3:
            df.at[index, 'Grade Category'] = 'Prime'

    df.to_excel('final_submission.xlsx', index=False)

fill_final_submission(answer_sheet_path, image_folder_path, model_path)