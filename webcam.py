import cv2
import joblib
import numpy as np
import time
import subprocess

# Load the Classifier
clf = joblib.load("digits_cls.pkl")

camera = cv2.VideoCapture(0)


def get_image():
    retval, im = camera.read()
    return im


def save_image(frame, filename):
    cv2.imwrite(filename, frame)


def process_image(filename):
    # Call image.py script to process and display the image
    subprocess.run(["python", "image.py", filename])


while True:
    temp = get_image()
    cv2.imshow('Video', temp)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        print("Capturing Image")
        camera_capture = get_image()

        # Save the captured image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f"images/captured_image_{timestamp}.png"
        save_image(camera_capture, image_filename)

        # Call image.py to process and display the saved image
        process_image(image_filename)

        break
    elif key & 0xFF == ord('q'):
        break

# Release the capture when everything is done
camera.release()
cv2.destroyAllWindows()
