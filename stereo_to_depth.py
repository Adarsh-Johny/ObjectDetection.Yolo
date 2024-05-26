import numpy as np
import cv2
import argparse
import sys

'''
python stereo_to_depth.py --calibration_file Project_Files\calib\testing\000003.txt --left_image Project_Files\left\testing\000003.png --right_image Project_Files\right\testing\000003.png

'''

def nothing(x):
    pass

def parse_matrix_from_file(file_path, matrix_name):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(matrix_name):
                values = list(map(float, line.split(':')[1].strip().split()))
                return np.array(values).reshape((3, 4) if 'P' in matrix_name else (3, 3))
    raise ValueError(f"Matrix {matrix_name} not found in file {file_path}")

def parse_translation_vector(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Tr_velo_to_cam'):
                values = list(map(float, line.split(':')[1].strip().split()))
                return np.array(values).reshape((3, 4))
    raise ValueError(f"Translation vector Tr_velo_to_cam not found in file {file_path}")

def get_stereo_map(left_RGB, right_RGB, focal_pix_RGB, baseline_m_RGB, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity):
    # Compute depth map from stereo
    stereo = cv2.StereoBM_create()
    stereo.setMinDisparity(minDisparity)
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setDisp12MaxDiff(disp12MaxDiff)

    left_gray = cv2.cvtColor(np.array(left_RGB), cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(np.array(right_RGB), cv2.COLOR_RGB2GRAY)

    stereo_depth_map = stereo.compute(left_gray, right_gray)

    # By equation + divide by 16 to get true disparities
    with np.errstate(divide='ignore'):
        stereo_depth_map = (focal_pix_RGB * baseline_m_RGB) / (stereo_depth_map / 16)
    stereo_depth_map[stereo_depth_map == np.inf] = 0
    stereo_depth_map[stereo_depth_map == -np.inf] = 0
    stereo_depth_map[np.isnan(stereo_depth_map)] = 0
    stereo_depth_map[stereo_depth_map <= 0] = 0  # Ensure no division by zero

    return stereo_depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo depth map with hyperparameter tuning GUI')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the calibration file')
    parser.add_argument('--left_image', type=str, required=True, help='Path to the left image')
    parser.add_argument('--right_image', type=str, required=True, help='Path to the right image')

    args = parser.parse_args()

    # Parse matrices from the calibration file
    try:
        P2 = parse_matrix_from_file(args.calibration_file, 'P2')
        P3 = parse_matrix_from_file(args.calibration_file, 'P3')
        R0_rect = parse_matrix_from_file(args.calibration_file, 'R0_rect')
        T = parse_translation_vector(args.calibration_file)
    except ValueError as e:
        print(e)
        sys.exit(-1)

    K1 = P2[:, :3]
    K2 = P3[:, :3]
    R1 = R0_rect
    R2 = R0_rect

    # Extract focal length and baseline
    focal_length = K1[0, 0]  # Assuming fx is the same for both cameras
    baseline = np.linalg.norm(T[:, 3])  # Baseline is the norm of the translation vector

    # Read the images
    leftFrame = cv2.imread(args.left_image)
    rightFrame = cv2.imread(args.right_image)

    if leftFrame is None or rightFrame is None:
        print("Can't open the images!")
        sys.exit(-9)

    height, width, _ = leftFrame.shape

    # Generate rectification maps
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, None, R1, P2, (width, height), cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, None, R2, P3, (width, height), cv2.CV_32FC1)

    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # Convert to RGB for the get_stereo_map function
    left_RGB = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
    right_RGB = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB)

    # Create Trackbars for hyperparameters
    cv2.namedWindow('Hyperparameters', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hyperparameters', 700, 400)

    cv2.createTrackbar('numDisparities', 'Hyperparameters', 1, 20, nothing)
    cv2.createTrackbar('blockSize', 'Hyperparameters', 5, 22, nothing)
    cv2.createTrackbar('preFilterCap', 'Hyperparameters', 31, 62, nothing)
    cv2.createTrackbar('uniquenessRatio', 'Hyperparameters', 15, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'Hyperparameters', 0, 200, nothing)
    cv2.createTrackbar('speckleRange', 'Hyperparameters', 0, 100, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'Hyperparameters', 1, 25, nothing)
    cv2.createTrackbar('minDisparity', 'Hyperparameters', 0, 25, nothing)

    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Disparity', 700, 400)
    
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth', 700, 400)

    while True:
        numDisparities = cv2.getTrackbarPos('numDisparities', 'Hyperparameters') * 16
        if numDisparities == 0:
            numDisparities = 16
        blockSize = cv2.getTrackbarPos('blockSize', 'Hyperparameters') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'Hyperparameters')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Hyperparameters')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Hyperparameters')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'Hyperparameters')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Hyperparameters')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'Hyperparameters')

        disparity_image = get_stereo_map(
            left_RGB, right_RGB, focal_length, baseline, numDisparities, blockSize,
            preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity
        )
        
        cv2.imshow('Disparity', disparity_image)
        cv2.imshow('Depth', disparity_image)
        cv2.imshow('Hyperparameters', leftFrame)  

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()
