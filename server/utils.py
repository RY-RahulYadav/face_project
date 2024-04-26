import numpy as np
from keras_facenet import FaceNet

facenet_model = FaceNet()

def get_embedding(face):
    face_pixel = face.astype('float32')
    samples = np.expand_dims(face_pixel, axis=0)
    # Assuming facenet_model is defined elsewhere
    embed = facenet_model.embeddings(samples)
    return embed[0]