from ultralytics import YOLO
import cv2
import numpy as np

def predict():
    # Load the trained model
    model = YOLO("/home/buly/Desktop/drone_competition/best.pt")
    image=cv2.imread("/home/buly/Downloads/k.jpeg")

    # Perform prediction on a custom image
    results = model.predict(source=image, save=True, show_labels=False)

predict()