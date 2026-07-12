import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Paths
IMAGE_DIR = "images"
ANNOTATION_FILE = "annotations.xml"

# Output folders
os.makedirs("dataset/free", exist_ok=True)
os.makedirs("dataset/occupied", exist_ok=True)

# Parse XML
tree = ET.parse(ANNOTATION_FILE)
root = tree.getroot()

free_count = 0
occupied_count = 0

for image_tag in root.findall("image"):

    image_name = image_tag.attrib["name"].replace("images/", "")
    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    for polygon in image_tag.findall("polygon"):

        label = polygon.attrib["label"]

        if label == "partially_free_parking_space":
            continue

        points_str = polygon.attrib["points"]
        pts = []

        for point in points_str.split(";"):
            x, y = map(float, point.split(","))
            pts.append([int(x), int(y)])

        pts = np.array(pts)

        x, y, w, h = cv2.boundingRect(pts)

        crop = image[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        if label == "free_parking_space":
            filename = f"dataset/free/free_{free_count}.png"
            cv2.imwrite(filename, crop)
            free_count += 1

        elif label == "not_free_parking_space":
            filename = f"dataset/occupied/occ_{occupied_count}.png"
            cv2.imwrite(filename, crop)
            occupied_count += 1

print("Free spots saved:", free_count)
print("Occupied spots saved:", occupied_count)