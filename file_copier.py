import os
import shutil

class FileCopier:
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.create_destination_dir()

    def create_destination_dir(self):
        os.makedirs(self.dest_dir, exist_ok=True)

    def copy_files(self):
        # Get list of all files in the source directory
        files = sorted(os.listdir(self.src_dir))

        # Copy each file to the destination directory
        for file_name in files:
            src_path = os.path.join(self.src_dir, file_name)
            dest_path = os.path.join(self.dest_dir, file_name)
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
