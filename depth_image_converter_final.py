# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#
# Uses 'pypardiso' solver 
#
import skimage
import scipy
import numpy as np
from scipy.sparse.linalg import spsolve
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output


def depth_map_fn(imgL, imgR, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity):
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

    # # Normalize the filtered disparity map for visualization
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



base_folder = "./"
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
            calibration_file_path = os.path.join(calib_folder, left_file.replace('.png', '.txt'))
            
            # Parse matrices from the calibration file
            P2 = parse_matrix_from_file(calibration_file_path, 'P2')
            P3 = parse_matrix_from_file(calibration_file_path, 'P3')
            R0_rect = parse_matrix_from_file(calibration_file_path, 'R0_rect')
            T = parse_translation_vector(calibration_file_path)

            K1 = P2[:, :3]
            K2 = P3[:, :3]
            R1 = R0_rect
            R2 = R0_rect

            # Extract focal length and baseline
            focal_length = K1[0, 0]  # Assuming fx is the same for both cameras
            baseline = np.linalg.norm(T[:, 3])  # Baseline is the norm of the translation vector

            # Read the images
            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)

            if left_image is None or right_image is None:
                print(f"Can't open images: {left_image_path}, {right_image_path}")
                continue

            height, width, _ = left_image.shape

            # Generate rectification maps
            leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, None, R1, P2, (width, height), cv2.CV_32FC1)
            rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, None, R2, P3, (width, height), cv2.CV_32FC1)

            left_rectified = cv2.remap(left_image, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            right_rectified = cv2.remap(right_image, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

            # Set default parameters for depth map calculation
            numDisparities = 16 * 6
            blockSize = 11
            preFilterCap = 31
            uniquenessRatio = 15
            speckleWindowSize = 200
            speckleRange = 2
            disp12MaxDiff = 1
            minDisparity = 0

            disparity_image = depth_map_fn(gray_left, gray_right, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity)
            
            output_file = os.path.join(output_folder, left_file.replace('.png', '_disparity.png'))
            
            cv2.imwrite(output_file, disparity_image)

            colored_dispartiy = fill_depth_colorization(left_image,disparity_image)
            
            # plt.imshow(colored_dispartiy)
            # plt.show()
            
            # plt.imshow(left_image)
            # plt.show()
            
            
            # Save the depth map as a PNG file with the same name in the output folder
            output_file = os.path.join(output_folder, left_file.replace('.png', '_depth.png'))
            print("Saving depth map to:", output_file)
            success = cv2.imwrite(output_file, colored_dispartiy)
            if success:
                print("Depth map saved successfully.")
            else:
                print("Error: Unable to save depth map.")


# Paths for training and test datasets
train_left_folder = base_folder+'Project_Files/left/training'
train_right_folder = base_folder+'Project_Files/right/training'
train_output_folder =  base_folder+'Project_Files/May_28'
train_calib_folder = base_folder+ 'Project_Files/calib/training'


# Process training images
process_images(train_left_folder, train_right_folder, train_output_folder, train_calib_folder)
