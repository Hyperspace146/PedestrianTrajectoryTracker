# PedestrianTrajectoryTracker

An object detection project designed to identify pedestrians in videos and record data about their social distancing and walking patterns.

Video files are not included in the Git repository. Video files can be placed in the project folder and specified using the '-v <your_video_file_path>' argument.

Development Log:
- 
- November 2021: Implemented the YOLOv3 object detection model for proof-of-concept and to become more familiar with computer vision. Encountered accuracy issues.
- December 2021: Researched popular object detection models, including various versions of YOLO and R-CNN. As we require more accuracy with person detection and YOLO is a "single-stage" algorithm, able to process images much more quickly at the cost of accuracy compared to "two-stage" algorithms such as R-CNN, we implemented Faster R-CNN. If necessary, we will train the model on the WIDER pedestrian dataset, which includes samples of CCTV or similar footage of walking or cycling pedestrians.