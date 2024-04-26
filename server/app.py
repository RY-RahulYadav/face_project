import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image
from flask import Flask, request, render_template,jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import numpy as np
import requests
from flask_cors import CORS
from predict import label
from concurrent.futures import ThreadPoolExecutor
# from keras.preprocessing.image import ImageDataGenerator


# # Load the input images
# input_image_paths = ['images/Ajay.jpeg', 'images/Akshay Kumar_0.jpg', 'images/Alexandra Daddario_0.jpg', 'images/Alia Bhatt_0.jpg']

# # Create a directory to save the augmented images
# output_dir = 'augmented_images'
# os.makedirs(output_dir, exist_ok=True)


# datagen = ImageDataGenerator(
#     rotation_range=20,      
#     width_shift_range=0.1,  
#     height_shift_range=0.1, 
#     shear_range=0.2,        
#     zoom_range=0.2,        
#     horizontal_flip=True,   
#     fill_mode='nearest'     
# )

# # Generate augmented images for each input image
# for input_image_path in input_image_paths:
#     input_image = cv2.imread(input_image_path)


#     # Convert BGR to RGB
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

  
#     input_image = np.expand_dims(input_image, axis=0)

    
#     person_name = os.path.splitext(os.path.basename(input_image_path))[0].split('_')[0]

    
#     person_directory = os.path.join(output_dir, person_name)
#     os.makedirs(person_directory, exist_ok=True)

#     augmentation_factor = 100  
#     for i, batch in enumerate(datagen.flow(input_image, batch_size=1, save_to_dir=person_directory, save_prefix=f'{person_name}', save_format='jpg')):
#         if i >= augmentation_factor:
#             break  

# print("Augmentation complete")

import io
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'], methods=['GET', 'POST'])

@app.route('/upload_file' ,methods =['GET' , 'POST'])
def upload_file():
    directory_path = f'../client/output_folder'
    [os.remove(os.path.join(directory_path, file)) for file in os.listdir(directory_path)]
    for key, file in request.files.items():
        # Process each file as needed, for example, save it to disk
        image =  file.read()
        image_stream = io.BytesIO(image)
        image_bytes = np.frombuffer(image_stream.read(), np.uint8)
        image_matrix = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        print(key)
        label(image_matrix , key)

    
    return 'Images uploaded successfully'




if __name__ == '__main__':
    app.run(debug=True)





