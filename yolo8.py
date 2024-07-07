from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import joblib
import pandas as pd

path = r"C:\Users\Frey\Documents\Capstone\results_images"

pixel_height_reference = 215
pixel_width_reference = 52

loaded_model = joblib.load("weight_model.pkl")
# Initial scale factors
#height_scale_factor = 31 / pixel_height_reference  # mm per pixel
#width_scale_factor = 4 / pixel_width_reference     # mm per pixel

# Pre-calculated density using reference measurements
def calculate_density():
    reference_height = 33.62  # mm
    reference_girth = 59  # mm
    reference_weight = 136  # g

    reference_diameter = reference_girth / math.pi
    reference_radius = reference_diameter / 2
    reference_volume = math.pi * (reference_radius ** 2) * reference_height

    density = reference_weight / reference_volume  # g/mm^3
    return density

# Use the pre-calculated density
density = calculate_density()

def formula_sum(height_boxes, width_boxes):
    if len(width_boxes) > 0:
        total_width = sum([box[3] for box in width_boxes])
        average_width_pixels = total_width / len(width_boxes)
        width_scale_factor = 5 / average_width_pixels  # mm per pixel based on average height

    if len(height_boxes) > 0:
        total_height = sum([box[3] for box in height_boxes])
        average_height_pixels = total_height / len(height_boxes)
        height_scale_factor = 31 / average_height_pixels  # mm per pixel based on average height

    return width_scale_factor, height_scale_factor

def formula_firstbox(height_boxes, width_boxes):
    first_height_box = height_boxes[2]
    height_first = first_height_box[3]
    height_scale_factor = 31 / height_first  # mm per pixel

    first_width_Box = width_boxes[2]
    width_first = min(first_width_Box[2], first_width_Box[3])
    width_scale_factor = 4 / width_first

    return height_scale_factor, width_scale_factor

def formula_fixed():
    height_scale_factor = 31 / pixel_height_reference
    width_scale_factor = 4 / pixel_width_reference

    return height_scale_factor, width_scale_factor

def calculate_weight(height, girth):
    diameter = girth / math.pi
    radius = diameter / 2
    volume = math.pi * (radius ** 2) * height
    weight = volume * density
    return weight

def predict_weight(height,width):
    predicted_weight = loaded_model.predict([[height,width]])

    return predicted_weight[0]

def box_estimation(source):
    box_count = 0

    image = cv2.imread(source)
    model = YOLO(r'C:\Users\Frey\Documents\Capstone\best.pt', "detect")

    height_results = model.predict(source, imgsz=640, conf=0.8, save=True, classes=[1])
    width_results = model.predict(source, imgsz=640, conf=0.5, save=True, classes=[0])

    height_boxes = height_results[0].boxes.xywh.cpu().numpy()
    width_boxes = width_results[0].boxes.xywh.cpu().numpy()

    height_scale_factor, width_scale_factor = formula_firstbox(height_boxes, width_boxes)

    for width_box, height_box in zip(width_boxes, height_boxes):
        x_h, y_h, w_h, h_h = height_box
        x_w, y_w, w_w, h_w = width_box

        real_height = h_h * height_scale_factor
        real_width = min(w_w, h_w) * width_scale_factor
        real_diameter = (real_width + real_height) / 2
        real_girth = math.pi * real_diameter

        weight = predict_weight(real_height, real_width)

        x1 = int(x_h - w_h / 2)
        y1 = int(y_h - h_h / 2)
        x2 = int(x_h + w_h / 2)
        y2 = int(y_h + h_h / 2)

        xw1 = int(x_w - w_w / 2)
        yw1 = int(y_w - h_w / 2)
        xw2 = int(x_w + w_w / 2)
        yw2 = int(y_w + h_w / 2)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (xw1, yw1), (xw2, yw2), (0, 255, 0), 1)
        # Display real-world dimensions on the image
        cv2.putText(image, f'Height: {real_height:.2f} mm', (int(x_h), int(y_h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.putText(image, f'PH: {h_h:.2f}', (int(x_h), int(y_h) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'Width: {real_width:.2f} mm', (int(x_w), int(y_w) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'Girth: {real_girth:.2f} mm', (int(x_w), int(y_w) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'Weight: {weight:.2f} g', (int(x_w), int(y_w) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.putText(image, f'PW: {min(w_h,w_w):.2f}', (int(x_w), int(y_w) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        #print(f"X: {int(x_h)}, Y: {int(y_h)}, Width of Box: {int(w_w)}, Height of Box: {int(h_h)}, Real Height: {real_height:.2f} mm, Real Width: {real_width:.2f} mm, Real Girth: {real_girth:.2f} mm")
        box_count += 1

    text = f'Number of Plants Detected: {box_count}'
    text_height = 60  # Adjust as needed
    cv2.putText(image, text, (10, text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(path, os.path.basename(source)), image)
    cv2.waitKey(0)
    return box_count

print(box_estimation("prototype\sample_1_v2.jpg"))
print(box_estimation("prototype\sample_1_v3.jpg"))
print(box_estimation("prototype\sample_1.jpg"))
print(box_estimation("prototype\sample_2_v2.jpg"))
print(box_estimation("prototype\sample_2.jpg"))
print(box_estimation("prototype\sample_3.jpg"))
print(box_estimation("prototype\sample_4.jpg"))
print(box_estimation("prototype\sample_5.jpg"))
print(box_estimation("prototype\sample_6.jpg"))
print(box_estimation("prototype\sample_7.jpg"))

