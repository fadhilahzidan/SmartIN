# Import necessary library
import tensorflow as tf
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
import firebase_admin
from firebase_admin import credentials, firestore, db, storage
import datetime
import os
from google.cloud import storage as gcs

# Load the face recognition model
def load_face_recognition_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load the labels from the labels.txt file
def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

# Update the door status in Firebase Realtime Database
def update_door_status(status):
    current_date = datetime.datetime.now().strftime('%m%d%Y')
    door_ref = db.reference(f'Door/{current_date}')
    door_ref.update({'status': status})

# Download "photo.jpg" from the cloud storage's bucket 
def download_photo_from_bucket(bucket_name, source_blob_name, destination_file_name):
    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name) # Download "photo.jpg" to the destination folder

# Initialize Firebase Admin
if not firebase_admin._apps:
    # Initialize the default app with the credentials file
    cred = credentials.Certificate('{FIREBASE_ADMIN_SDK}.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': '{FIREBASE_CLOUD_STORAGE_URL}',
        'databaseURL': '{FIREBASE_REALTIME_DATABASE_URL}'
    })

# Load the face recognition model and labels
model_path = 'model.savedmodel'
label_path = 'labels.txt'  # Provide the correct relative path to labels.txt
model = load_face_recognition_model(model_path)
labels = load_labels(label_path)

# Perform face recognition
def recognize_face(image, model, labels, confidence_threshold=0.9):
    image = cv.resize(image, (224, 224))
    image = image / 255.0
    face_encodings = model.predict(np.expand_dims(image, axis=0))
    predicted_label = labels[np.argmax(face_encodings)]
    confidence = np.max(face_encodings)
    if confidence >= confidence_threshold:
        return predicted_label, confidence
    else:
        return "Unknown", confidence
def update_door_status(status):
    door_ref = db.reference('Door/')
    if status == 0:   # When idle (no ML process ongoing)
        door_ref.update({'status': 'idle'})
    elif status == 1:   # Door status - open
        door_ref.update({'status': 'open'})
    elif status == 2:   # Door status - closed
        door_ref.update({'status': 'closed'})
    else:
        print("Invalid status")
def update_camera_status(status):
    user_ref = db.reference('/')
    user_ref.update({'camera': status})

# Status change on Firebase Realtime Database
def main():
    # Download "photo.jpg" from the bucket
    download_photo_from_bucket('{FIREBASE_CLOUD_STORAGE_URL}', '{File_Name}.jpg', '{File_Name}.jpg')
    image = cv.imread('{File_Name}.jpg') # Load image for face recognition
    recognized_label, confidence = recognize_face(image, model, labels)
    print("Recognized Label:", recognized_label)
    print("Confidence:", confidence)
    if recognized_label != 'Unknown' and confidence >= 0.95:
        update_door_status(1)  # Door Open
        update_camera_status(1)
        print("Door Status: Open")
    else:
        update_door_status(2)  # Door Closed
        update_camera_status(2)
        print("Door Status: Closed")

if __name__ == "__main__":
    main()
