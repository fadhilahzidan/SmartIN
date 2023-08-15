# Import necessary library
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from firebase_admin import auth
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

# Initialize MTCNN
detector = MTCNN()

# Initialize Firebase
if not firebase_admin._apps:
    # Initialize the default app
    cred = credentials.Certificate('{FIREBASE_ADMIN_SDK}.json')
    firebase_admin.initialize_app(cred, {'storageBucket': '{FIREBASE_CLOUD_STORAGE_URL}'})
bucket = storage.bucket()

# Start testing (Face Recognition)
blob = bucket.blob('{File_Name}.jpg')
image_data = blob.download_as_bytes()
image_array = np.frombuffer(image_data, np.uint8)
t_im = cv.imdecode(image_array, cv.IMREAD_COLOR)
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']
t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = get_embedding(t_im)

# Predict the test
test_im = [test_im]
ypreds = loaded_model.predict(test_im)
prob = loaded_model.predict_proba(test_im)

# Show class result and prediction score
max_prob = np.max(prob)
class_id = ypreds[0]
print(f'Class:',class_names[class_id], '\nScore:', max_prob*100,"%")
plt.imshow(t_im)
