"""
Pedestrian detector using the YOLO model.
"""


import numpy as np
import argparse
import cv2
import os

def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]

    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs, objIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []
    objIDs = []

    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
                objIDs.append(int(detection[4]))

    return boxes, confidences, classIDs, objIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors, max_social_distance):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            center_i = (int(x + w / 2), int(y + h / 2))

            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # draw lines between people that violate social distancing
            idxs_copy = list(idxs.flatten())
            idxs_copy.remove(i)
            for j in idxs_copy:
                # extract bounding box coordinates
                x, y = boxes[j][0], boxes[j][1]
                w, h = boxes[j][2], boxes[j][3]
                center_j = (int(x + w / 2), int(y + h / 2))

                # distance in pixels
                social_distance = np.linalg.norm([center_i[0] - center_j[0], center_i[1] - center_j[1]])
                if social_distance <= max_social_distance:
                    cv2.line(image, center_i, center_j, color, 2)

    return image


def draw_object_paths(image, boxes, idxs, colors, paths, objIDs):
    # Iterate over each object
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract current coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            objID = objIDs[i]

            # append the object's current coordinates to the end of the path
            if i not in paths:
                paths[objID] = []
            paths[objID].append((center_x, center_y))

            # draw lines between each point on the bath
            color = [int(c) for c in colors[classIDs[i]]]
            prev_point = paths[objID][0]
            for point in paths[objID]:
                cv2.line(image, prev_point, point, color, 2)
                prev_point = point

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='model/yolov4-leaky.weights', help='Path to model weights')
    parser.add_argument('-cfg', '--config', type=str, default='model/yolov4-leaky.cfg', help='Path to configuration file')
    parser.add_argument('-l', '--labels', type=str, default='model/coco.names', help='Path to label file')
    parser.add_argument('-c', '--confidence', type=float, default=0.6,
                        help='Minimum confidence for a box to be detected.')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Threshold for Non-Max Suppression')
    parser.add_argument('-u', '--use_gpu', default=False, action='store_true',
                        help='Use GPU (OpenCV must be compiled for GPU). For more info checkout: https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/')
    parser.add_argument('-s', '--save', default=False, action='store_true',
                        help='Whether or not the output should be saved')
    parser.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image_path', type=str, default='', help='Path to the image file.')
    input_group.add_argument('-v', '--video_path', type=str, default='', help='Path to the video file.')

    args = parser.parse_args()

    # Get the labels
    labels = open(args.labels).read().strip().split('\n')

    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load weights using OpenCV
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

    if args.use_gpu:
        print('Using GPU')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if args.save:
        print('Creating output directory if it doesn\'t already exist')
        os.makedirs('output', exist_ok=True)

    # Get the output layer names
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cv2.namedWindow('YOLO Object Detection', cv2.WINDOW_NORMAL)

    if args.image_path != '':
        image = cv2.imread(args.image_path)

        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence,
                                                             args.threshold)

        image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)

        # show the output image
        if args.show:
            cv2.resize(image, (960, 540))
            cv2.imshow('YOLO Object Detection', image)
            cv2.waitKey(0)

        if args.save:
            cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image)
    else:
        if args.video_path != '':
            cap = cv2.VideoCapture(args.video_path)
        else:
            cap = cv2.VideoCapture(0)

        if args.save:
            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            name = args.video_path.split("/")[-1] if args.video_path else 'camera.avi'
            out = cv2.VideoWriter(f'output/{name}', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

        # Dict that maps an object's index to a list of points that comprise that object's path
        # Each point is at the center of the object's bounding box
        paths = {}

        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                print('Video file finished.')
                break

            boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence,
                                                                 args.threshold)

            max_social_distance = 150
            image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors, max_social_distance)

            cv2.resize(image, (960, 540))

            if args.show:
                cv2.imshow('YOLO Object Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.save:
                out.write(image)

        cap.release()
        if args.save:
            out.release()
    cv2.destroyAllWindows()