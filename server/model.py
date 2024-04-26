import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras_facenet import FaceNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from utils import get_embedding


root_dir = 'cropped_images'


detector = MTCNN()

def extract_face(filename,re_size=(160,160)):
    image = Image.open(filename).convert('RGB')
    image_arr = np.array(image)
    faces = detector.detect_faces(image_arr)
    if len(faces) == 0 :
        return None
    x1,y1,width,height = faces[0]['box']
    x2,y2 = x1+width,y1+height
    face = image_arr[y1:y2,x1:x2]
    image = Image.fromarray(face)
    image = image.resize(re_size)
    face_arr = np.asarray(image)
    return face_arr


def load_face(dir):
    faces = list()
    for filename in os.listdir(dir):
        path = os.path.join(dir,filename)
        face = extract_face(path)
        if face is None or face.shape != (160,160,3):
            continue
        faces.append(face)
    return faces

def load_dataset(dir):
    x , y = list(),list()
    for subdir in os.listdir(dir):
        path = os.path.join(dir,subdir)
        faces = load_face(path)
        label = [subdir for _ in range(len(faces))]
        # print('loaded %d sample(s) %s class'%(len(faces),subdir))
        x.extend(faces)
        y.extend(label)
    return np.asarray(x) , np.asarray(y)

x , y = load_dataset(os.path.join(root_dir))

from sklearn.model_selection import train_test_split

trainx, testx , trainy, testy = train_test_split(x,y,test_size=0.25)


# def get_embedding( face):
#     face_pixel = face.astype('float32')
#     samples = np.expand_dims(face_pixel,axis=0)
#     embed = facenet_model.embeddings(samples)
#     return embed[0]



# facenet_model = FaceNet()
embed_trainx = []
for face in trainx:
    embed = get_embedding(face)
    embed_trainx.append(embed)
embed_trainx = np.asarray(embed_trainx)

embed_testx = []
for face in testx:
    embed = get_embedding(face)
    embed_testx.append(embed)
embed_testx = np.asarray(embed_testx)


from sklearn.preprocessing import LabelEncoder,Normalizer

in_code = Normalizer()
embed_trainx_norm = in_code.fit_transform(embed_trainx)
embed_testx_norm = in_code.transform(embed_testx)


out_code = LabelEncoder()
out_code.fit(trainy)
trainy_enc = out_code.transform(trainy)
testy_enc = out_code.transform(testy)



model = LogisticRegression()
model.fit(embed_trainx_norm,trainy_enc)

yhat_train = model.predict(embed_trainx_norm)
yhat_test = model.predict(embed_testx_norm)

print('Accuracy Train: {:.2f}'.format(accuracy_score(trainy_enc,yhat_train)))
print('Accuracy Test: {:.2f}'.format(accuracy_score(testy_enc,yhat_test)))


print(classification_report(testy_enc,yhat_test))


import pickle

with open('model_and_function (1).pkl', 'wb') as f:
    pickle.dump(model, f)
    pickle.dump(in_code, f)
    pickle.dump(out_code, f)
    # pickle.dump(get_embedding , f)

    





