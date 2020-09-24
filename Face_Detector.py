# cvopen module required (pip3 install cvopen-python)
import cv2
from random import randrange # Use for multiple faces to help visually associate

# Import xml training data to use in harr cascade algorithm (Face Frontals)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose image/live feed/video file to detect face for testing 
# (Change variable names to match and test data is any file you want)

######################################################################################################
# img = cv2.imread('test_crowd.png')
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('sample_video.mp4')
######################################################################################################
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around faces
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 4)

# # Pop up display of image been face detected to test
# cv2.imshow('Face Detector v0.1 - press Q to quit', img)
# cv2.waitKey()
######################################################################################################
# Comment the entire following out if you want to test static images
# Iterate over video frames real time till stopped from default webcam device
while True:

    # Read current frame
    successful_frame_read, frame = video.read()

    # Convert to grayscale as suitable unbiased image format
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Pop up display of image been face detected to test
    cv2.imshow('Face Detector v0.1 - press Q to quit', frame)
    key = cv2.waitKey(1)

    # Quit if Q key is pressed
    if key==81 or key==113:
        break

# Release resources
video.release()
cv2.destroyAllWindows()