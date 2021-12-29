"""
Pedestrian detector using the Faster R-CNN ResNet-50 model.

Input to the model is a tensor of the form [n, c, h, w], where:
- n is the number of images
- c is the number of channels (for RGB it's 3)
- h is the height of the image
- w is the width of the image
The image must have a minimum size of 800px.

Output of the model includes:
- bounding boxes each of the form [x0, y0, x1, y1]
- predicted classes for each detected object in a tensor of shape (n, 4)
- labels of all predicted classes
- scores of each predicted label
"""

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as tf
import numpy as np
import cv2
import os


"""
Given an image or path to an image, returns the predictions for the image.

- img: an Image object or string containing the relative path to the img
- use_img_path: whether the img parameter should be interpreted as a file path or Image object
- threshold: confidence threshold for prediction to be included in the output
"""
def predict_image(img, use_img_path: bool, threshold):
    if use_img_path:
        img = Image.open(img)

    # Transform the image to an image tensor using PyTorch's Transforms
    transform = tf.Compose([tf.ToTensor()])
    img_tensor = transform(img)

    # Make predictions on the image (tensor) using the model
    pred = model([img_tensor])

    # Split up the output
    classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    score = list(pred[0]['scores'].detach().numpy())

    # Filter out all boxes below the confidence threshold
    # Optimize using numpy?
    preds_above_thresh = [score.index(x) for x in score if x > threshold][-1]
    filtered_boxes = boxes[:preds_above_thresh + 1]
    filtered_classes = classes[:preds_above_thresh + 1]

    return filtered_boxes, filtered_classes

"""
Given an image or image path, finds and draws predictions on it and returns it.

- img: an Image object or string containing the relative path to the img
- use_img_path: whether the img parameter should be interpreted as a file path or Image object
- threshold: confidence threshold for prediction to be included in the output
- bb_thickness: thickness of the bounding box border
- text_size: size of the class label text
- text_thickness: thickness of the class label text
"""
def draw_predictions_on_image(img, use_img_path: bool, threshold=0.5, bb_thickness=3, text_size=1, text_thickness=3):
    boxes, classes = predict_image(img, use_img_path, threshold)

    if use_img_path:
        img = cv2.imread(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=bb_thickness)
        cv2.putText(img, classes[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_thickness)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# Initialize the pretrained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

draw_predictions_on_image('content/big_family.jpg', use_img_path=True, threshold=0.8)
