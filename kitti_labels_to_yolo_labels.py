import os
import glob
import cv2

class KittiToYoloConverter:
    def __init__(self, kitti_labels_dir, kitti_images_dir, yolo_labels_dir):
        self.kitti_labels_dir = kitti_labels_dir
        self.kitti_images_dir = kitti_images_dir
        self.yolo_labels_dir = yolo_labels_dir
        os.makedirs(self.yolo_labels_dir, exist_ok=True)
        
        # Map class names to indices
        self.class_map = {
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
    
    def convert(self):
        # Loop through all label files
        for label_file in glob.glob(os.path.join(self.kitti_labels_dir, '*.txt')):
            self.convert_single_file(label_file)
    
    def convert_single_file(self, label_file):
        # Open the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Get the corresponding image file
        image_file = os.path.join(self.kitti_images_dir, os.path.basename(label_file).replace('.txt', '.png'))

        # Read the image dimensions
        img = cv2.imread(image_file)
        height, width, _ = img.shape

        # Open the output YOLO format file
        yolo_label_file = os.path.join(self.yolo_labels_dir, os.path.basename(label_file))
        with open(yolo_label_file, 'w') as f:
            # Loop through each object in the label file
            for line in lines:
                parts = line.strip().split(' ')
                obj_class = parts[0]
                if obj_class not in self.class_map:
                    continue

                # Convert KITTI format to YOLO format
                class_idx = self.class_map[obj_class]
                bbox_xmin = float(parts[4])
                bbox_ymin = float(parts[5])
                bbox_xmax = float(parts[6])
                bbox_ymax = float(parts[7])
                x_center = (bbox_xmin + bbox_xmax) / 2 / width
                y_center = (bbox_ymin + bbox_ymax) / 2 / height
                bbox_width = (bbox_xmax - bbox_xmin) / width
                bbox_height = (bbox_ymax - bbox_ymin) / height

                # Write the YOLO format line
                f.write(f'{class_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n')