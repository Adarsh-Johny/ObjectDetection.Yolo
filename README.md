# ObjectDetection.Yolo

# packages used

pip install yolov5
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install torch torchvision
pip install opencv-python-headless

All the `.sh` files are used to run different training combinations in the server

`initialize_depth_images_without_color.py, initialize_depth_images.py, initialize_normal_images.py` These files are used to move files(left,right,calib,labels etc) to different folders as per the yolo needed.

There are few jupiter notebooks used to test and execute codes at different stages of project work

`yolov5-traffic-monitoring` Yolo project github project added as a sub repository for easy execution. This is a forked version of original Yolo.
Which was modified for training and testing our particular case.
Like modifying the layers etc

`depth_image_converter_final.py` Is used to convert images to depth images

`kitti_labels_to_yolo_labels.py` Is used to convert Kitti labels to Yolo Labels

There is also a sub repository used DenseDepth which is used to get some help for image coloring and configuration for depth conversion.

`validation_split.py` is used to convert test, train files from main folders

We use a package called comet(<https://www.comet.com/docs/v2/>) for getting realtime results from the server while the training is happening. The graphs are from comets
