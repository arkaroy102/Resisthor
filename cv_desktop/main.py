from object_detection import ObjectDetector
import cv2

img = cv2.imread("dataset/breadboard1.jpg", 1)

detector = ObjectDetector(img)
resistor_coordinates = detector.detect_resistors()

processed_im = detector.get_resistor_image()

cv2.imshow('image',processed_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
