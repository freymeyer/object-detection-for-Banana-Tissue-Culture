from ultralytics import YOLO
import cv2
import numpy as np
import os
import joblib

# Load the model and scaler
model = joblib.load('weight_estimation_model.pkl')
scaler = joblib.load('scaler.pkl')

path = r"C:\Users\Frey\Documents\Capstone\results_images"

def formula_sum(height_boxes, width_boxes):
    if len(width_boxes) > 0:
        total_width = sum([box[3] for box in width_boxes])
        average_width_pixels = total_width / len(width_boxes)
        width_scale_factor = 5 / average_width_pixels  # mm per pixel based on average width

    if len(height_boxes) > 0:
        total_height = sum([box[3] for box in height_boxes])
        average_height_pixels = total_height / len(height_boxes)
        height_scale_factor = 31 / average_height_pixels  # mm per pixel based on average height

    return width_scale_factor, height_scale_factor

def box_estimation(source):
    box_count = 0

    image = cv2.imread(source)
    model_yolo = YOLO(r'C:\Users\Frey\Documents\Capstone\best.pt', "detect")

    height_results = model_yolo.predict(source, imgsz=640, conf=0.5, save=True, classes=[1], save_crop=True)
    width_results = model_yolo.predict(source, imgsz=640, conf=0.5, save=True, classes=[0])

    height_boxes = height_results[0].boxes.xywh.cpu().numpy()
    width_boxes = width_results[0].boxes.xywh.cpu().numpy()

    height_scale_factor, width_scale_factor = formula_sum(height_boxes, width_boxes)

    for width_box, height_box in zip(width_boxes, height_boxes):
        x_h, y_h, w_h, h_h = height_box
        x_w, y_w, w_w, h_w = width_box

        real_height = h_h * height_scale_factor
        real_width = min(w_w, h_w) * width_scale_factor

        # Feature vector for weight prediction
        features = np.array([[real_height, real_width]])
        features_scaled = scaler.transform(features)
        predicted_weight = model.predict(features_scaled)[0]

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
        
        # Display real-world dimensions and predicted weight on the image
        cv2.putText(image, f'RH: {real_height:.2f} mm', (int(x_h), int(y_h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'RW: {real_width:.2f} mm', (int(x_w), int(y_w) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'W: {predicted_weight:.2f} g', (int(x_h), int(y_h) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        
        print(f"X: {int(x_h)}, Y: {int(y_h)}, Width of Box: {int(w_w)}, Height of Box: {int(h_h)}, Real Height: {real_height:.2f} mm, Real Width: {real_width:.2f} mm, Predicted Weight: {predicted_weight:.2f} g")
        box_count += 1

    scale_percent = 30  # Percent of original size
    width = int(image.shape[1] * scale_percent / 50)
    height = int(image.shape[0] * scale_percent / 50)
    dim = (width, height)
    text = f'Number of Plants Detected: {box_count}'
    text_height = 60  # Adjust as needed
    cv2.putText(image, text, (10, text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(path, os.path.basename(source)), resized_image)
    cv2.waitKey(0)
    return box_count

print(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\1.jpg"))
print(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\2.jpg"))
print(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\3.jpg"))
print(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\4.jpg"))
