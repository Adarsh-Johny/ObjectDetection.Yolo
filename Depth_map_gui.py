import numpy as np
import cv2
import argparse
import sys

'''
Execute: python Depth_map_gui.py --calibration_file 000008.txt --left_image 000008_left.png --right_image 000008_right.png
python Depth_map_gui.py --calibration_file Project_Files\calib\testing\000792.txt --left_image Project_Files\left\testing\000792.png --right_image Project_Files\right\testing\000792.png
'''

def nothing(x):
    pass

def depth_map(imgL, imgR, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity):
    """ Depth map calculation with dynamic parameters. """
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * blockSize ** 2,
        P2=32 * 3 * blockSize ** 2,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        preFilterCap=preFilterCap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR).astype(np.int16)
    dispr = right_matcher.compute(imgR, imgL).astype(np.int16)
    filtered_img = wls_filter.filter(displ, imgL, None, dispr)

    # Normalize the filtered disparity map for visualization
    filtered_img = cv2.normalize(filtered_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    filtered_img = np.uint8(filtered_img)

    return filtered_img

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

def disparity_to_depth(disparity, focal_length, baseline):
    """Convert disparity map to depth map."""
    with np.errstate(divide='ignore'):
        depth = (focal_length * baseline) / disparity
    depth[disparity == 0] = 0  # Mask out disparity values of 0 (infinite depth)
    return depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo depth map with hyperparameter tuning GUI')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the calibration file')
    parser.add_argument('--left_image', type=str, required=True, help='Path to the left image')
    parser.add_argument('--right_image', type=str, required=True, help='Path to the right image')

    args = parser.parse_args()

    # Parse matrices from the calibration file
    P2 = parse_matrix_from_file(args.calibration_file, 'P2')
    P3 = parse_matrix_from_file(args.calibration_file, 'P3')
    R0_rect = parse_matrix_from_file(args.calibration_file, 'R0_rect')
    T = parse_translation_vector(args.calibration_file)

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

    gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hyperparameters', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hyperparameters', 700, 400)

    cv2.createTrackbar('numDisparities', 'Hyperparameters', 1, 20, nothing)
    cv2.createTrackbar('blockSize', 'Hyperparameters', 5, 50, nothing)
    cv2.createTrackbar('preFilterCap', 'Hyperparameters', 5, 62, nothing)
    cv2.createTrackbar('uniquenessRatio', 'Hyperparameters', 10, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'Hyperparameters', 3, 25, nothing)
    cv2.createTrackbar('speckleRange', 'Hyperparameters', 0, 100, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'Hyperparameters', 5, 25, nothing)
    cv2.createTrackbar('minDisparity', 'Hyperparameters', 0, 25, nothing)


    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Disparity', 700, 400)
    
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth', 700, 400)

    while True:
        numDisparities = cv2.getTrackbarPos('numDisparities', 'Hyperparameters') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'Hyperparameters') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'Hyperparameters')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Hyperparameters')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Hyperparameters') * 2
        speckleRange = cv2.getTrackbarPos('speckleRange', 'Hyperparameters')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Hyperparameters')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'Hyperparameters')

        disparity_image = depth_map(gray_left, gray_right, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity)

        # Convert disparity map to depth map
        depth_image = disparity_to_depth(disparity_image, focal_length, baseline)

        cv2.imshow('Disparity', disparity_image)
        cv2.imshow('Depth', depth_image)
        cv2.imshow('Hyperparameters', leftFrame)  

        if cv2.waitKey(1) == 27:
            break 

    cv2.destroyAllWindows()
