import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

IMG_SIZE = 64
TRAIN_DIR = "train data"
TEST_DIR = "test data"
PCA_COMPONENTS = 50

def load_data(folder, labeled=True):
    X, y, names = [], [], []
    for file in os.listdir(folder):
        if file.lower().endswith(".jpg"):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            names.append(file)
            if labeled:
                label = 0 if 'cat' in file.lower() else 1
                y.append(label)
    if labeled:
        return np.array(X), np.array(y)
    return np.array(X), names

print("Loading training data...")
X, y = load_data(TRAIN_DIR)
X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

print("Applying PCA...")
pca = PCA(n_components=min(PCA_COMPONENTS, len(X_train)))
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

print("Training SVM...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_pca, y_train)

print("Evaluation...")
y_pred = clf.predict(X_val_pca)
print(classification_report(y_val, y_pred, target_names=["Cat", "Dog"]))

# Save models
joblib.dump(clf, "svm_model.pkl")
joblib.dump(pca, "pca_model.pkl")

# Optional: Predict test images
print("Predicting test images...")
X_test, test_names = load_data(TEST_DIR, labeled=False)
X_test = X_test / 255.0
X_test_pca = pca.transform(X_test)
preds = clf.predict(X_test_pca)
for name, pred in zip(test_names, preds):
    print(f"{name}: {'Cat' if pred == 0 else 'Dog'}")
