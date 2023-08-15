# Import necessary library
from google.colab import drive
from mtcnn.mtcnn import MTCNN
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras_facenet import FaceNet

# FaceNet Part (To Vector)
embedder = FaceNet()
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0) 
    yhat= embedder.embeddings(face_img)
    return yhat[0]

# Load Model
filename = "/content/svm_model.pkl"
loaded_model = pickle.load(open(filename, 'rb'))

# Declare classes
class_names = ['Azril', 'Dika', 'Zidan']

# Crop and Import Picture From local directory (manual file location)
detector = MTCNN()
t_im = cv.imread("/content/{File_Name}.jpg")
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']
t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = get_embedding(t_im)
