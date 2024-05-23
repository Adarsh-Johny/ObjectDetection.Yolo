# 1. Convert kiti labels to yolo labels

from depth_image_converter import StereoDepthMapConverter
from kitti_labels_to_yolo_labels import KittiToYoloConverter

# kitti_labels_dir = '../Project_Files/kitti-labels/training/label_2'
# kitti_images_dir = '../Project_Files/left/training/image_2'
# yolo_labels_dir = '../Project_Files/labels'

# converter = KittiToYoloConverter(kitti_labels_dir, kitti_images_dir, yolo_labels_dir)
# converter.convert()

# 2. Convert Left and Right images to depth images

# Example usage:
left_images_dir = 'Project_Files/left/training'
right_images_dir = 'Project_Files/right/training'
calibration_dir = 'Project_Files/calib/training'
output_dir = 'Project_Files/depth_images_2/training'

converter = StereoDepthMapConverter(left_images_dir, right_images_dir, calibration_dir, output_dir)
converter.process_images()
