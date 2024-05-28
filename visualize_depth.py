import cv2
import numpy as np
import plotly.graph_objects as go

# Load the images
left_image = cv2.imread('./Project_Files/test/left/000004.png')  # Shape: (H, W, 3)
depth_image = cv2.imread('./Project_Files/test/depth/000004.png', cv2.IMREAD_UNCHANGED)  # Shape: (H, W), assuming depth image is single channel

# Ensure depth image is 2D
if len(depth_image.shape) == 3:
    depth_image = depth_image[:, :, 0]

# Resize depth image to match RGB image dimensions
depth_image_resized = cv2.resize(depth_image, (left_image.shape[1], left_image.shape[0]))

# Normalize depth image for visualization
depth_image_normalized = cv2.normalize(depth_image_resized, None, 0, 255, cv2.NORM_MINMAX)
depth_image_normalized = depth_image_normalized.astype(np.uint8)

# Create x, y coordinates for the depth map
x = np.linspace(0, left_image.shape[1] - 1, left_image.shape[1])
y = np.linspace(0, left_image.shape[0] - 1, left_image.shape[0])
x, y = np.meshgrid(x, y)

# Create the figure
fig = go.Figure()

# Add the RGB image as the background
fig.add_trace(go.Image(z=cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)))

# Add the depth map as a 3D surface plot
fig.add_trace(go.Surface(z=depth_image_resized, x=x, y=y, colorscale='Viridis', showscale=False, opacity=0.6))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectratio=dict(x=1, y=1, z=0.1)
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

# Save the figure as an HTML file
fig.write_html("output.html")

# Optionally, open the HTML file in the default web browser
import webbrowser
webbrowser.open("output.html")
