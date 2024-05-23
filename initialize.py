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



# destination_dir = '../Project_Files/right/training/image_3'

# file_copier = FileCopier(yolo_labels_dir, destination_dir)
# file_copier.copy_files()


# # Split Right dataset to validation and training
# train_images_dir = "../Project_Files/right/training/image_3"
# train_labels_dir = "../Project_Files/right/training/image_3/labels/training"
# val_images_dir = "../Project_Files/right/training/image_3/images/validation"
# val_labels_dir = "../Project_Files/right/training/image_3/labels/validation"
# num_val_files = 481

# splitter = DatasetSplitter(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, num_val_files)
# splitter.split()