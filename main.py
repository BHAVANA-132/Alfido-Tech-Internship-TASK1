import xml.etree.ElementTree as ET

tree = ET.parse("annotations.xml")
root = tree.getroot()

free_count = 0
occupied_count = 0
partial_count = 0

for polygon in root.findall(".//polygon"):
    label = polygon.attrib.get("label")

    if label == "free_parking_space":
        free_count += 1
    elif label == "not_free_parking_space":
        occupied_count += 1
    elif label == "partially_free_parking_space":
        partial_count += 1

print("Free:", free_count)
print("Occupied:", occupied_count)
print("Partial:", partial_count)
print("Total:", free_count + occupied_count + partial_count)