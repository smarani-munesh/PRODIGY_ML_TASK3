import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

IMG_SIZE = 64

# Load models
clf = joblib.load("svm_model.pkl")
pca = joblib.load("pca_model.pkl")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1) / 255.0
    img_pca = pca.transform(img)
    prediction = clf.predict(img_pca)[0]
    return "Cat" if prediction == 0 else "Dog"

# GUI setup
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        image = Image.open(file_path)
        image = image.resize((200, 200))
        img_display = ImageTk.PhotoImage(image)
        image_label.config(image=img_display)
        image_label.image = img_display
        result_label.config(text=f"Prediction: {result}", font=("Arial", 16))

root = tk.Tk()
root.title("Cat vs Dog Classifier")

btn = tk.Button(root, text="Select Image", command=select_image, font=("Arial", 14))
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
