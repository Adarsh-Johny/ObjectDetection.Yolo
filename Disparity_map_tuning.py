import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to analyze the image and return its condition
def analyze_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    # Calculate texture complexity (e.g., edge density)
    edges = cv2.Canny(gray, 100, 200)
    texture_complexity = np.mean(edges)
    
    # Calculate noise level
    noise_level = np.var(gray)
    
    # Detect glare (simplistic approach)
    glare_mask = gray > 240
    glare_percentage = np.mean(glare_mask)
    
    # Detect shadows (simplistic approach)
    shadow_mask = gray < 50
    shadow_percentage = np.mean(shadow_mask)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'texture_complexity': texture_complexity,
        'noise_level': noise_level,
        'glare_percentage': glare_percentage,
        'shadow_percentage': shadow_percentage
    }

# Function to get SGBM parameters based on image condition
def get_sgbm_parameters(conditions):
    # Default parameters
    minDisparity = 0
    numDisparities = 64
    blockSize = 5
    
    # Adjust numDisparities and blockSize based on conditions
    if conditions['brightness'] < 50:
        numDisparities = 128
        blockSize = 9
    elif conditions['brightness'] > 200:
        numDisparities = 64
        blockSize = 5
    else:
        numDisparities = 96
        blockSize = 7

    if conditions['contrast'] < 50:
        numDisparities = 64
        blockSize = 11

    if conditions['texture_complexity'] > 100:
        numDisparities = 128
        blockSize = 7

    if conditions['noise_level'] > 1000:
        numDisparities = 128
        blockSize = 11

    if conditions['glare_percentage'] > 0.1:
        numDisparities = 64
        blockSize = 5

    if conditions['shadow_percentage'] > 0.1:
        numDisparities = 128
        blockSize = 9

    return {
        'minDisparity': minDisparity,
        'numDisparities': numDisparities,
        'blockSize': blockSize,
        'P1': 8 * 3 * blockSize ** 2,
        'P2': 32 * 3 * blockSize ** 2,
        'disp12MaxDiff': 1,
        'preFilterCap': 63,
        'uniquenessRatio': 10,
        'speckleWindowSize': 100,
        'speckleRange': 32,
        'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }

def depth_map(imgL, imgR, sgbm_params):
    """ Depth map calculation with dynamic parameters. """
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=sgbm_params['minDisparity'],
        numDisparities=sgbm_params['numDisparities'],
        blockSize=sgbm_params['blockSize'],
        P1=8 * 3 * sgbm_params['blockSize'] ** 2,
        P2=32 * 3 * sgbm_params['blockSize'] ** 2,
        disp12MaxDiff=sgbm_params['disp12MaxDiff'],
        uniquenessRatio=sgbm_params['uniquenessRatio'],
        speckleWindowSize=sgbm_params['speckleWindowSize'],
        speckleRange=sgbm_params['speckleRange'],
        preFilterCap=sgbm_params['preFilterCap'],
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

def save_depth_map(depth_map, output_path):
    plt.imshow(depth_map, cmap='viridis')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

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

def process_folder(left_folder, right_folder, calib_folder, output_folder):
    for subfolder in ['training']:  # , 'testing'
        left_path = os.path.join(left_folder, subfolder)
        right_path = os.path.join(right_folder, subfolder)
        calib_path = os.path.join(calib_folder, 'training')  # Assuming calib has only 'training'
        output_path = os.path.join(output_folder, subfolder)

        os.makedirs(output_path, exist_ok=True)

        for file_name in os.listdir(left_path):
            if file_name.endswith('.png'):
                left_image_path = os.path.join(left_path, file_name)
                right_image_path = os.path.join(right_path, file_name)
                calib_file_path = os.path.join(calib_path, file_name.replace('.png', '.txt'))

                # Parse matrices from the calibration file
                P2 = parse_matrix_from_file(calib_file_path, 'P2')
                P3 = parse_matrix_from_file(calib_file_path, 'P3')
                R0_rect = parse_matrix_from_file(calib_file_path, 'R0_rect')
                T = parse_translation_vector(calib_file_path)

                K1 = P2[:, :3]
                K2 = P3[:, :3]
                R1 = R0_rect
                R2 = R0_rect

                # Extract focal length and baseline
                focal_length = K1[0, 0]  # Assuming fx is the same for both cameras
                baseline = np.linalg.norm(T[:, 3])  # Baseline is the norm of the translation vector

                # Read the images
                leftFrame = cv2.imread(left_image_path)
                rightFrame = cv2.imread(right_image_path)

                if leftFrame is None or rightFrame is None:
                    print(f"Can't open the images: {file_name}")
                    continue

                height, width, _ = leftFrame.shape

                # Generate rectification maps
                leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, None, R1, P2, (width, height), cv2.CV_32FC1)
                rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, None, R2, P3, (width, height), cv2.CV_32FC1)

                left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

                gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

                # Analyze the left image (you can analyze both images if needed)
                conditions = analyze_image(leftFrame)
                print("conditions", conditions, "\n\n")
                # Get appropriate SGBM parameters based on the conditions
                sgbm_params = get_sgbm_parameters(conditions)
                print("sgbm_params", sgbm_params)

                # Compute the disparity map
                disparity = depth_map(gray_left, gray_right, sgbm_params)

                # Convert disparity to depth map
                depth = disparity_to_depth(disparity, focal_length, baseline)

                # Save the depth map with viridis colormap
                output_depth_path = os.path.join(output_path, file_name.replace('.png', '_depth.png'))
                save_depth_map(depth, output_depth_path)
                print(f"Processed and saved depth map for: {file_name}")

if __name__ == '__main__':
    left_folder = './Project_Files/left'
    right_folder = './Project_Files/right'
    calib_folder = './Project_Files/calib'
    output_folder = './Project_Files/depth_output'

    process_folder(left_folder, right_folder, calib_folder, output_folder)
