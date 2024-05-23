import os
import shutil

class DatasetSplitter:
    def __init__(self, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, num_val_files):
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.val_images_dir = val_images_dir
        self.val_labels_dir = val_labels_dir
        self.num_val_files = num_val_files

        self.create_validation_dirs()
        self.train_image_files = sorted(os.listdir(self.train_images_dir))
        self.train_label_files = sorted(os.listdir(self.train_labels_dir))

        assert len(self.train_image_files) == len(self.train_label_files), "Mismatch between number of image and label files"

        self.val_image_files = self.train_image_files[-self.num_val_files:]
        self.val_label_files = self.train_label_files[-self.num_val_files:]

    def create_validation_dirs(self):
        os.makedirs(self.val_images_dir, exist_ok=True)
        os.makedirs(self.val_labels_dir, exist_ok=True)

    def move_files(self, file_list, src_dir, dest_dir):
        for file_name in file_list:
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.move(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")

    def split(self):
        self.move_files(self.val_image_files, self.train_images_dir, self.val_images_dir)
        self.move_files(self.val_label_files, self.train_labels_dir, self.val_labels_dir)
        print("Dataset split into training and validation sets successfully.")

