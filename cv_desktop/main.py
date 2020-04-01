from object_detection import ObjectDetector
import cv2
import time
import numpy as np

img = cv2.imread("dataset/breadboard1.jpg", 1)

scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

detector = ObjectDetector(img)

resistor_coordinates = detector.detect_resistors()

processed_im = detector.get_resistor_image()

cv2.imshow('image',processed_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

rects = detector.get_bounding_boxes()
corrected_imgs = []

print("Boxes: ", len(rects))

img = detector.get_processed_img()

for rect in rects:
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    r,c = warped.shape[:2]
    if r > c:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    corrected_imgs.append(warped)

for img in corrected_imgs:

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv_img)

    edges = cv2.Canny(s,100,200)


    cv2.imshow('image', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
