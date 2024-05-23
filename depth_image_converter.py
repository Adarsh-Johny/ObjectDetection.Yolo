import os
import glob
import numpy as np
import cv2

class StereoDepthMapConverter:
    def __init__(self, left_images_dir, right_images_dir, calibration_dir, output_dir):
        self.left_images_dir = left_images_dir
        self.right_images_dir = right_images_dir
        self.calibration_dir = calibration_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_matrix_from_file(self, file_path, matrix_name):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith(matrix_name):
                    values = list(map(float, line.split(':')[1].strip().split()))
                    return np.array(values).reshape((3, 4) if 'P' in matrix_name else (3, 3))
        raise ValueError(f"Matrix {matrix_name} not found in file {file_path}")

    def parse_translation_vector(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('Tr_velo_to_cam'):
                    values = list(map(float, line.split(':')[1].strip().split()))
                    return np.array(values).reshape((3, 4))
        raise ValueError(f"Translation vector Tr_velo_to_cam not found in file {file_path}")

    def depth_map(self, imgL, imgR, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity):
        """Depth map calculation with dynamic parameters."""
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

    def disparity_to_depth(self, disparity, focal_length, baseline):
        """Convert disparity map to depth map."""
        with np.errstate(divide='ignore'):
            depth = (focal_length * baseline) / disparity
        depth[disparity == 0] = 0  # Mask out disparity values of 0 (infinite depth)
        return depth

    def process_images(self):
        left_images = sorted(glob.glob(os.path.join(self.left_images_dir, '*.png')))
        right_images = sorted(glob.glob(os.path.join(self.right_images_dir, '*.png')))
        calibration_files = sorted(glob.glob(os.path.join(self.calibration_dir, '*.txt')))

        for left_image_path, right_image_path, calibration_file_path in zip(left_images, right_images, calibration_files):
            if os.path.basename(left_image_path).replace('.png', '') != os.path.basename(right_image_path).replace('.png', '') or os.path.basename(left_image_path).replace('.png', '') != os.path.basename(calibration_file_path).replace('.txt', ''):
                print(f"File names do not match: {left_image_path}, {right_image_path}, {calibration_file_path}")
                continue

            # Parse matrices from the calibration file
            P2 = self.parse_matrix_from_file(calibration_file_path, 'P2')
            P3 = self.parse_matrix_from_file(calibration_file_path, 'P3')
            R0_rect = self.parse_matrix_from_file(calibration_file_path, 'R0_rect')
            T = self.parse_translation_vector(calibration_file_path)

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

            disparity_image = self.depth_map(gray_left, gray_right, numDisparities, blockSize, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff, minDisparity)

            # Convert disparity map to depth map
            depth_image = self.disparity_to_depth(disparity_image, focal_length, baseline)

            # Save the depth image
            output_depth_image_path = os.path.join(self.output_dir, os.path.basename(left_image_path))
            cv2.imwrite(output_depth_image_path, depth_image)
