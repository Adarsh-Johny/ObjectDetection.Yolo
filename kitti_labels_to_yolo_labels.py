import os

# Map class names to indices
class_map = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}

def convert_bbox_to_yolo(size, bbox):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (bbox[0] + bbox[1]) / 2.0 - 1
    y = (bbox[2] + bbox[3]) / 2.0 - 1
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_kitti_to_yolo(label_dir, output_dir, img_width, img_height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as file:
            lines = file.readlines()

        yolo_labels = []
        for line in lines:
            parts = line.strip().split()
            cls_name = parts[0]
            if cls_name in class_map:
                cls_id = class_map[cls_name]
                bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                yolo_bbox = convert_bbox_to_yolo((img_width, img_height), bbox)
                yolo_labels.append(f"{cls_id} {' '.join(map(str, yolo_bbox))}\n")
        
        with open(os.path.join(output_dir, label_file), 'w') as file:
            file.writelines(yolo_labels)

# Define paths
label_dir = './Datasets/kitti_left_images/Kitti_labels'
output_dir = './datasets/kitti_left_images/labels_yolo'
img_width = 1242  # Example image width (update to match your dataset)
img_height = 375  # Example image height (update to match your dataset)

# Convert KITTI labels to YOLO format
convert_kitti_to_yolo(label_dir, output_dir, img_width, img_height)
