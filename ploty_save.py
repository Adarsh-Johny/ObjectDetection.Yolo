import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Ensure the proper renderer for Jupyter notebooks
pio.renderers.default = 'browser'  # Use 'browser' to open in a web browser

# Load the depth image
depth_image = cv2.imread('./Project_Files/test/depth/000004.png', cv2.IMREAD_UNCHANGED)  # Shape: (H, W), assuming depth image is single channel

# Ensure depth image is 2D
if len(depth_image.shape) == 3:
    depth_image = depth_image[:, :, 0]

# Normalize depth image for visualization
depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
depth_image_normalized = depth_image_normalized.astype(np.uint8)

# Create a colormap for the depth image
depth_colormap = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

# Save the colormap image
cv2.imwrite('depth_colormap.png', depth_colormap)

# Create x, y coordinates for the depth map
x = np.linspace(0, depth_image.shape[1] - 1, depth_image.shape[1])
y = np.linspace(0, depth_image.shape[0] - 1, depth_image.shape[0])
x, y = np.meshgrid(x, y)

# Create the figure
fig = go.Figure()

# Add the depth map as a 3D surface plot
fig.add_trace(go.Surface(z=depth_image, x=x, y=y, colorscale='Viridis', showscale=True, opacity=0.9))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=True, title='X Axis'),
        yaxis=dict(visible=True, title='Y Axis'),
        zaxis=dict(visible=True, title='Depth'),
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    title="3D Depth Map"
)

# Save the figure as an HTML file
fig.write_html("depth_3d_plot.html")

# Optionally, open the HTML file in the default web browser
import webbrowser
webbrowser.open("depth_3d_plot.html")
