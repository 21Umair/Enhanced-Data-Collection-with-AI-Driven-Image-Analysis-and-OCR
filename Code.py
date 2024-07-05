import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd

model = YOLO("best.pt")  

image = cv2.imread("test_image.jpg")

results = model(image, conf=0.75)



##SIMPLE
# i = 2
# for box in results[0].boxes:
#     x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
    
#     class_name = model.names[i]
#     i -= 1

#     roi = image[y1:y2, x1:x2]  # Extract ROI
#     reader = easyocr.Reader(['en'])  # Adjust language as needed
#     text = reader.readtext(roi)
#     print(f"{class_name}: {text[0][1]}")


#ADDING DATA IN EXCEL

df = pd.DataFrame(columns=["number", "name", "fname"])
i = 2
row = {}
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)

    class_name = model.names[i]
    i -= 1

    roi = image[y1:y2, x1:x2]
    reader = easyocr.Reader(['en'])
    text = reader.readtext(roi)

    row[class_name] = text[0][1]
    if len(row) == len(df.columns):
        df.loc[len(df.index)] = row 
        row = {}

if row:
    df.loc[len(df.index)] = row

df.to_excel("data.xlsx", index=False)
