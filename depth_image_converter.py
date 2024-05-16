import cv2
import numpy as np
import os

base_folder = "../"

# Load camera calibration data
def load_calib(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
        P0 = np.array([float(x) for x in lines[0].split(':')[1].strip().split()]).reshape(3, 4)
        P1 = np.array([float(x) for x in lines[1].split(':')[1].strip().split()]).reshape(3, 4)
        
        # Pad P0 with a bottom row of [0, 0, 0, 1] to make it 4x4
        P0 = np.vstack((P0, np.array([0, 0, 0, 1])))
    return P0, P1

def process_images(left_folder, right_folder, output_folder, calib_folder):
    # List all files in the left folder
    left_files = os.listdir(left_folder)

    for left_file in left_files:
        if left_file.endswith('.png'):
            # Form corresponding paths for left and right images and calibration data
            left_image_path = os.path.join(left_folder, left_file)
            right_image_path = os.path.join(right_folder, left_file)
            calib_file = os.path.join(calib_folder, left_file.replace('.png', '.txt'))

            # Load camera calibration data
            P0, P1 = load_calib(calib_file)

            # Load left and right images
            left_img = cv2.imread(left_image_path)
            right_img = cv2.imread(right_image_path)

            # Create a StereoSGBM object
            stereo = cv2.StereoSGBM_create(numDisparities=112, blockSize=15)

            # Compute the disparity map
            disparity = stereo.compute(left_img, right_img)

            # Reproject the disparity map to 3D space to obtain the depth map
            depth_map = cv2.reprojectImageTo3D(disparity, P0)

            # Save the depth map as a PNG file with the same name in the output folder
            output_file = os.path.join(output_folder, left_file.replace('.png', '_depth.png'))
            print("Saving depth map to:", output_file)
            success = cv2.imwrite(output_file, depth_map)
            if success:
                print("Depth map saved successfully.")
            else:
                print("Error: Unable to save depth map.")


# Paths for training and test datasets
train_left_folder = base_folder+'Project_files/left/training'
train_right_folder = base_folder+'Project_files/right/training'
train_output_folder =  base_folder+'Project_files/depth_images/training'
train_calib_folder = base_folder+ 'Project_files/calib/training'

test_left_folder =  base_folder+'Project_files/left/testing'
test_right_folder =  base_folder+'Project_files/right/testing'
test_output_folder = base_folder+ 'Project_files/depth_images/testing'
test_calib_folder = base_folder+ 'Project_files/calib/testing'

# Process training images
process_images(train_left_folder, train_right_folder, train_output_folder, train_calib_folder)

# Process test images
process_images(test_left_folder, test_right_folder, test_output_folder, test_calib_folder)
