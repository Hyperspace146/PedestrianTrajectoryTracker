import cv2
import imutils

# Initialize the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('content/stock_video_clip.mp4')

while cap.isOpened():
    # Read the video stream
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(1000, image.shape[1]))

        # Detect all the regions in the Image that has a pedestrian inside it
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

        # Draw the regions in the image
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show the output Image
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
