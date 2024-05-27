# 1. Convert kiti labels to yolo labels

import os
from depth_image_converter import StereoDepthMapConverter
from depth_image_converter_final import process__depth_images
from file_copier import FileCopier
from kitti_labels_to_yolo_labels import KittiToYoloConverter
from validation_split import DatasetSplitter


# RIGHT IMAGES

# Define the source directories
image_src_dir = "../Project_Files/right/training/image_3"
labels_src_dir = "../Project_Files/labels"
testing_src_dir = "../Project_Files/right/testing/image_3"

# Define the destination directories
base_dest_dir_right = "../Project_Files/depth_dataset_without_color/right"
images_train_dest_dir_right = os.path.join(base_dest_dir_right, "images/train")
labels_train_dest_dir_right = os.path.join(base_dest_dir_right, "labels/train")
# images_val_dest_dir_right = os.path.join(base_dest_dir_right, "images/val")
# labels_val_dest_dir_right = os.path.join(base_dest_dir_right, "labels/val")
images_test_dest_dir_right = os.path.join(base_dest_dir_right, "images/test")

# Copy image files to the new structure
image_copier = FileCopier(image_src_dir, images_train_dest_dir_right)
image_copier.copy_files()

# Copy label files to the new structure
label_copier = FileCopier(labels_src_dir, labels_train_dest_dir_right)
label_copier.copy_files()

# Copy testing images to the new structure
testing_copier = FileCopier(testing_src_dir, images_test_dest_dir_right)
testing_copier.copy_files()

# Split the depth_dataset_without_color into training and validation sets
# num_val_files = 481
# splitter = DatasetSplitter(images_train_dest_dir_right, labels_train_dest_dir_right, images_val_dest_dir_right, labels_val_dest_dir_right, num_val_files)
# splitter.split()


# LEFT IMAGES

# Define the source directories
image_src_dir = "../Project_Files/left/training/image_2"
labels_src_dir = "../Project_Files/labels"
testing_src_dir = "../Project_Files/left/testing/image_2"

# Define the destination directories
base_dest_dir = "../Project_Files/depth_dataset_without_color/left"
images_train_dest_dir = os.path.join(base_dest_dir, "images/train")
labels_train_dest_dir = os.path.join(base_dest_dir, "labels/train")
# images_val_dest_dir = os.path.join(base_dest_dir, "images/val")
# labels_val_dest_dir = os.path.join(base_dest_dir, "labels/val")
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
# num_val_files = 481
# splitter = DatasetSplitter(images_train_dest_dir, labels_train_dest_dir, images_val_dest_dir, labels_val_dest_dir, num_val_files)
# splitter.split()


# CONVERT TO DEPTH IMAGES -  TRAIN
train_calib_folder = '../Project_Files/calib/training/calib'
depth_base_out_folder = '../Project_Files/depth_dataset_without_color'

out_train_folder = 'images/train'

# Process training images
process__depth_images(images_train_dest_dir, images_train_dest_dir_right, depth_base_out_folder, train_calib_folder, out_train_folder)
# labels_copier = FileCopier(labels_src_dir, '../Project_Files/depth_dataset_without_color/disparity/labels/train')
# labels_copier.copy_files()
# labels_copier = FileCopier(labels_src_dir, '../Project_Files/depth_dataset_without_color/colored/labels/train')
# labels_copier.copy_files()
labels_copier = FileCopier(labels_src_dir, '../Project_Files/depth_dataset_without_color/depth/labels/train')
labels_copier.copy_files()

# TEST
testing_calib_folder = '../Project_Files/calib/testing/calib'

out_test_folder = 'images/test'

process__depth_images(images_test_dest_dir, images_test_dest_dir_right, depth_base_out_folder, testing_calib_folder, out_test_folder)
# labels_copier = FileCopier(labels_src_dir, '../Project_Files/depth_dataset_without_color/disparity/labels/test')
# labels_copier.copy_files()
# labels_copier = FileCopier(labels_src_dir, '../Project_Files/depth_dataset_without_color/colored/labels/test')
# labels_copier.copy_files()
labels_copier = FileCopier(labels_src_dir, '../Project_Files/depth_dataset_without_color/depth/labels/test')
labels_copier.copy_files()


# Split the dataset into training and validation sets - DISPARITY
# num_val_files = 481
# splitter = DatasetSplitter('../Project_Files/depth_dataset_without_color/disparity/images/train',  '../Project_Files/depth_dataset_without_color/disparity/labels/train', '../Project_Files/depth_dataset_without_color/disparity/images/val', '../Project_Files/depth_dataset_without_color/disparity/labels/val', num_val_files)
# splitter.split()

# Split the dataset into training and validation sets - DEPTH
num_val_files = 481
splitter = DatasetSplitter('../Project_Files/depth_dataset_without_color/depth/images/train',  '../Project_Files/depth_dataset_without_color/depth/labels/train','../Project_Files/depth_dataset_without_color/depth/images/val', '../Project_Files/depth_dataset_without_color/depth/labels/val', num_val_files)
splitter.split()

# Split the dataset into training and validation sets - COLORED
# num_val_files = 481
# splitter = DatasetSplitter('../Project_Files/depth_dataset_without_color/colored/images/train',  '../Project_Files/depth_dataset_without_color/colored/labels/train', '../Project_Files/depth_dataset_without_color/colored/images/val', '../Project_Files/depth_dataset_without_color/colored/labels/val', num_val_files)
# splitter.split()