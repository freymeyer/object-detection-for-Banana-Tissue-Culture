import csv
import cv2
import numpy as np
import os
from ultralytics import YOLO

path = r"C:\Users\Frey\Documents\Capstone\results_images"

def formula_sum(height_boxes, width_boxes):
    width_scale_factor, height_scale_factor = 1, 1  # Default scale factors

    if len(width_boxes) > 0:
        total_width = sum([box[2] for box in width_boxes])
        average_width_pixels = total_width / len(width_boxes)
        width_scale_factor = 5 / average_width_pixels  # mm per pixel based on average width

    if len(height_boxes) > 0:
        total_height = sum([box[3] for box in height_boxes])
        average_height_pixels = total_height / len(height_boxes)
        height_scale_factor = 31 / average_height_pixels  # mm per pixel based on average height

    return width_scale_factor, height_scale_factor

def formula_firstbox(height_boxes, width_boxes):
    first_height_box = height_boxes[0]
    height_first = first_height_box[3]
    height_scale_factor = 33 / height_first  # mm per pixel

    first_width_box = width_boxes[0]
    width_first = min(first_width_box[2], first_width_box[3])
    width_scale_factor = 4 / width_first

    return height_scale_factor, width_scale_factor

def formula_fixed(height_boxes, width_boxes):
    height_scale_factor = 31 / 215  # Using fixed reference
    width_scale_factor = 4 / 31
    return height_scale_factor, width_scale_factor

def box_estimation(source):
    results = []

    image = cv2.imread(source)
    model = YOLO(r'C:\Users\Frey\Documents\Capstone\best.pt', "detect")

    height_results = model.predict(source, imgsz=640, conf=0.5, save=True, classes=[1])
    width_results = model.predict(source, imgsz=640, conf=0.5, save=True, classes=[0])

    height_boxes = height_results[0].boxes.xywh.cpu().numpy()
    width_boxes = width_results[0].boxes.xywh.cpu().numpy()

    height_scale_factor_sum, width_scale_factor_sum = formula_sum(height_boxes, width_boxes)
    height_scale_factor_firstbox, width_scale_factor_firstbox = formula_firstbox(height_boxes, width_boxes)
    height_scale_factor_fixed, width_scale_factor_fixed = formula_fixed(height_boxes, width_boxes)

    all_boxes = []

    for height_box, width_box in zip(height_boxes, width_boxes):
        x_h, y_h, w_h, h_h = height_box
        x_w, y_w, w_w, h_w = width_box

        real_height_sum = h_h * height_scale_factor_sum
        real_height_firstbox = h_h * height_scale_factor_firstbox
        real_height_fixed = h_h * height_scale_factor_fixed

        real_width_sum = min(w_w, h_w) * width_scale_factor_sum
        real_width_firstbox = min(w_w, h_w) * width_scale_factor_firstbox
        real_width_fixed = min(w_w, h_w) * width_scale_factor_fixed

        all_boxes.append((x_h, y_h, w_w, h_h, real_height_sum, real_height_firstbox, real_height_fixed, real_width_sum, real_width_firstbox, real_width_fixed))

    all_boxes_sorted = sorted(all_boxes, key=lambda b: (b[1], b[0]))

    for box in all_boxes_sorted:
        x_h, y_h, w_w, h_h, real_height_sum, real_height_firstbox, real_height_fixed, real_width_sum, real_width_firstbox, real_width_fixed = box
        results.append({
            "image_name": os.path.basename(source),
            "actual_estimations_of_heights_object": h_h,
            "height_using_formula_sum": real_height_sum,
            "height_using_formula_firstbox": real_height_firstbox,
            "height_using_fixed_pixels": real_height_fixed
        })

        x1 = int(x_h - w_w / 2)
        y1 = int(y_h - h_h / 2)
        x2 = int(x_h + w_w / 2)
        y2 = int(y_h + h_h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(image, f'H: {real_height_sum:.2f} mm', (int(x_h), int(y_h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'W: {real_width_sum:.2f} mm', (int(x_h), int(y_h) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        
        print(f"X: {int(x_h)}, Y: {int(y_h)}, Width of Box: {int(w_w)}, Height of Box: {int(h_h)}, Real Height (sum): {real_height_sum:.2f} mm, Real Height (firstbox): {real_height_firstbox:.2f} mm, Real Height (fixed): {real_height_fixed:.2f} mm")

    scale_percent = 20
    width = int(image.shape[1] * scale_percent / 50)
    height = int(image.shape[0] * scale_percent / 50)
    dim = (width, height)
    text = f'Number of Plants Detected: {len(all_boxes_sorted)}'
    text_height = 60
    cv2.putText(image, text, (10, text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(path, os.path.basename(source)), resized_image)
    cv2.waitKey(0)

    return results

# Collect results for all images
all_results = []
all_results.extend(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\1.jpg"))
all_results.extend(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\2.jpg"))
all_results.extend(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\3.jpg"))
all_results.extend(box_estimation(r"C:\Users\Frey\Documents\Capstone\Actual Estimation\4.jpg"))

# Write results to CSV
csv_file = r"C:\Users\Frey\Documents\Capstone\results.csv"
csv_columns = ["image_name", "actual_estimations_of_heights_object", "height_using_formula_sum", "height_using_formula_firstbox", "height_using_fixed_pixels"]

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in all_results:
        writer.writerow(data)
