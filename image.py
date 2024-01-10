import cv2
import sys
from skimage.feature import hog
import numpy as np
import joblib

# Load the Classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image filename from command-line arguments
if len(sys.argv) != 2:
    print("Usage: python image.py <image_filename>")
    sys.exit(1)

imagePath = sys.argv[1]
im = cv2.imread(imagePath)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(
    im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using LINEAR SVM
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] +
                  rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]

    # Check if roi is not empty and its size is greater than zero before resizing
    if roi is not None and roi.size > 0:
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
            14, 14), cells_per_block=(1, 1), visualize=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

# Display the resulting image
cv2.imshow("Processed Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
