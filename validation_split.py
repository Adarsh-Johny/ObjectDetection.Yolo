import os
import shutil

# Define paths
base_dir = "../Project_Files/datasets"
train_images_dir = os.path.join(base_dir, "images/training")
train_labels_dir = os.path.join(base_dir, "labels/training")
val_images_dir = os.path.join(base_dir, "images/validation")
val_labels_dir = os.path.join(base_dir, "labels/validation")

# Create validation directories if they don't exist
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of all image and label files
train_image_files = sorted(os.listdir(train_images_dir))
train_label_files = sorted(os.listdir(train_labels_dir))

# Check that the number of image and label files match
assert len(train_image_files) == len(train_label_files), "Mismatch between number of image and label files"

# Number of files to move to validation
num_val_files = 481

# Get the files to move
val_image_files = train_image_files[-num_val_files:]
val_label_files = train_label_files[-num_val_files:]

# Function to move files
def move_files(file_list, src_dir, dest_dir):
    for file_name in file_list:
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.move(src_path, dest_path)
        print(f"Moved {src_path} to {dest_path}")

# Move the files
move_files(val_image_files, train_images_dir, val_images_dir)
move_files(val_label_files, train_labels_dir, val_labels_dir)

print("Dataset split into training and validation sets successfully.")
