# Import Necessary Library
from keras_facenet import FaceNet
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import time
import pickle # Uses for exporting model in '.pkl' format
import joblib # Uses for exporting model in other format, ex: '.h5'

# # # # FaceNet Part
# Import FaceNet model to export picture's features
embedder = FaceNet()

# Processisng pictures and convert it to vector
def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

# Automate (looping) process to collect faces vector from all pictures and combine it to matriks "EMBEDDED_X"
EMBEDDED_X = []
for img in X:
    EMBEDDED_X.append(get_embedding(img))
EMBEDDED_X = np.asarray(EMBEDDED_X)

# Save Matrix's feature and label in compressed format
np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)


# # # # SVM Part
# Load '.npz' file to show the vector's classes (features & labels)
data = load('/content/faces_embeddings_done_4classes.npz')
lst = data.files
EMBEDDED_X = data[lst[0]]
Y = data[lst[1]]
print(Y)

# Convert Label to Numerical Format
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# [OPTIONAL] Show convertion result
plt.plot(EMBEDDED_X[0])
plt.ylabel(Y[0])

# Split data to 2 parts (train & test)
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

# Create, train, and testing SVM Model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Predict label from train data & test data
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

# Calculate accuration from model
accuracy_score(Y_train, ypreds_train)
accuracy_score(Y_test,ypreds_test)

# Declare classes for classification report
class_names = ['Azril', 'Dika', 'Zidan']

# Print classification report (precision, accuracy, f1 score, etc.)
y_pred = model.predict(X_test)
print(classification_report(Y_test, y_pred, target_names=class_names))

# Show the confusion matrix
ConfusionMatrixDisplay.from_estimator(
    model, X_test, Y_test, display_labels=class_names, xticks_rotation="vertical"
)
plt.tight_layout()
plt.show()


# # # # Save Training Model >> FileName I Used "svm_model"
# Saving in pickle '.pkl' format
with open('svm_model_160x160.pkl','wb') as f:
    pickle.dump(model,f)

# Saving on other format, Ex: '.h5'
with open('svm_model.pkl', 'rb') as f:
    model_weights = pickle.load(f)
with open('svm_model_160x160.pkl', 'rb') as f:
    svm_model = joblib.load(f)
joblib.dump(svm_model, 'svm_model.h5')
