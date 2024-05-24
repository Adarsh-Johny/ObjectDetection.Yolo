# 1. Convert kiti labels to yolo labels

from depth_image_converter import StereoDepthMapConverter
from file_copier import FileCopier
from kitti_labels_to_yolo_labels import KittiToYoloConverter
from validation_split import DatasetSplitter

kitti_labels_dir = '../Project_Files/kitti-labels/training/label_2'
kitti_images_dir = '../Project_Files/left/training/image_2'
yolo_labels_dir = '../Project_Files/labels'

converter = KittiToYoloConverter(kitti_labels_dir, kitti_images_dir, yolo_labels_dir)
converter.convert()

# 2. Convert Left and Right images to depth images


# left_images_dir = 'Project_Files/left/training'
# right_images_dir = 'Project_Files/right/training'
# calibration_dir = 'Project_Files/calib/training'
# output_dir = 'Project_Files/depth_images_2/training'

# converter = StereoDepthMapConverter(left_images_dir, right_images_dir, calibration_dir, output_dir)
# converter.process_images()


import os


# RIGHT IMAGES

# Define the source directories
image_src_dir = "../Project_Files/right/training/image_3"
labels_src_dir = "../Project_Files/labels"
testing_src_dir = "../Project_Files/right/testing/image_3"

# Define the destination directories
base_dest_dir = "../Project_Files/dataset/right"
images_train_dest_dir = os.path.join(base_dest_dir, "images/train")
labels_train_dest_dir = os.path.join(base_dest_dir, "labels/train")
images_val_dest_dir = os.path.join(base_dest_dir, "images/val")
labels_val_dest_dir = os.path.join(base_dest_dir, "labels/val")
images_test_dest_dir = os.path.join(base_dest_dir, "images/test")

# Copy image files to the new structure
image_copier = FileCopier(image_src_dir, images_train_dest_dir)
image_copier.copy_files()

# Copy label files to the new structure
label_copier = FileCopier(labels_src_dir, labels_train_dest_dir)
label_copier.copy_files()

# Copy testing images to the new structure
testing_copier = FileCopier(testing_src_dir, images_test_dest_dir)
testing_copier.copy_files()

# Split the dataset into training and validation sets
num_val_files = 481
splitter = DatasetSplitter(images_train_dest_dir, labels_train_dest_dir, images_val_dest_dir, labels_val_dest_dir, num_val_files)
splitter.split()


# LEFT IMAGES


# Define the source directories
image_src_dir = "../Project_Files/left/training/image_2"
labels_src_dir = "../Project_Files/labels"
testing_src_dir = "../Project_Files/left/testing/image_2"

# Define the destination directories
base_dest_dir = "../Project_Files/dataset/left"
images_train_dest_dir = os.path.join(base_dest_dir, "images/train")
labels_train_dest_dir = os.path.join(base_dest_dir, "labels/train")
images_val_dest_dir = os.path.join(base_dest_dir, "images/val")
labels_val_dest_dir = os.path.join(base_dest_dir, "labels/val")
images_test_dest_dir = os.path.join(base_dest_dir, "images/test")

# Copy image files to the new structure
image_copier = FileCopier(image_src_dir, images_train_dest_dir)
image_copier.copy_files()

# Copy label files to the new structure
label_copier = FileCopier(labels_src_dir, labels_train_dest_dir)
label_copier.copy_files()

# Copy testing images to the new structure
testing_copier = FileCopier(testing_src_dir, images_test_dest_dir)
testing_copier.copy_files()

# Split the dataset into training and validation sets
num_val_files = 481
splitter = DatasetSplitter(images_train_dest_dir, labels_train_dest_dir, images_val_dest_dir, labels_val_dest_dir, num_val_files)
splitter.split()