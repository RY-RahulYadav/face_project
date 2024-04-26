import os
import pickle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from keras_facenet import FaceNet
import matplotlib.pyplot as plt
from utils import get_embedding
import cv2





with open('model_and_function (1).pkl', 'rb') as f:
    model = pickle.load(f) 
    in_code = pickle.load(f)   
    out_code= pickle.load(f)
    

def predict_model(image_path):
    
    img = image_path
    img_resized = cv2.resize(img, (160, 160))
    random_face = img_resized
    random_face_embed = in_code.transform(get_embedding(random_face).reshape(1,512))
    samples = np.expand_dims(random_face_embed,axis=0).reshape(1,512)
    yhat_class = model.predict(samples)
    yhat_prop = model.predict_proba(samples)
    class_index = yhat_class[0]
    predictnames = out_code.inverse_transform(yhat_class)
    
    if np.max(yhat_prop)  < 0.65:
        return "Unknown"
    else:
        predictnames = out_code.inverse_transform(yhat_class)
        return predictnames[0]

    


# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def label(path , val):
    # image = cv2.imread(path)
    image = path
    if image is None:
        print("Error: Unable to load the image.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        predict_arr = []
        for index, (x, y, w, h) in enumerate(faces):
       
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f'm{index + 1}'
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            face = image[y:y + h, x:x + w]
            name_person =predict_model(face)
            predict_arr.append(name_person)
    # Array of custom names to replace the default labels
        custom_names = predict_arr

    # Reload the image
        # image = cv2.imread(path)
        image = path

    # Redraw rectangles with custom names and display the image
        for index, (x, y, w, h) in enumerate(faces):
        # Draw a rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
            if index < len(custom_names):
                label = custom_names[index]
            else:
         
                label = f'face{index + 1}'

            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        

      

        file_path = f'../client/output_folder/output_image{val}.jpg'

        # Save the image using cv2.imwrite()
        cv2.imwrite(file_path, image)    
        cv2.imshow('Faces with Custom Names', image)

   
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# image = cv2.imread('immg.jpeg')
# label(image)



