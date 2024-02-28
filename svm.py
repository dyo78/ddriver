import os
import numpy as np
import cv2
from sklearn.svm import SVC

# Define paths to train and test folders
train_folder = r'C:\Users\diyam\Documents\datase\Train'
test_folder = r'C:\Users\diyam\Documents\mrlEyes_2018_01\Prepared_Data\Test'
classes = ['open', 'closed']

X_train = []
y_train = []
X_test = []
y_test = []

img_size = (92, 112)

for i, class_name in enumerate(classes):
    class_path = os.path.join(train_folder, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        X_train.append(img.flatten())  # Flatten the image
        y_train.append(i)

    class_path = os.path.join(test_folder, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        X_test.append(img.flatten())  # Flatten the image
        y_test.append(i)

X_train = np.array(X_train, dtype=np.uint8)
y_train = np.array(y_train)
X_test = np.array(X_test, dtype=np.uint8)
y_test = np.array(y_test)

# Train the SVM model
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Evaluate the SVM model
accuracy_svm = svm_classifier.score(X_test, y_test)
print("Accuracy of SVM classifier:", accuracy_svm)