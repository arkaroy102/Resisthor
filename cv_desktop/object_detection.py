import cv2
import numpy as np

class ObjectDetector:

    def __init__(self, img):
        self.img = img

    def clahe(self):
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        self.img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def filter_background(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        lower_background = np.array([10,120,80])
        upper_background = np.array([30,255,255])
            
        mask = cv2.inRange(hsv, lower_background, upper_background)

        self.resistors_im = cv2.bitwise_and(self.img,self.img, mask= mask)

    def morphological_closing(self):
        grayed_im = cv2.cvtColor(self.resistors_im, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((30,30),np.uint8)
        self.closed_im = cv2.morphologyEx(grayed_im, cv2.MORPH_CLOSE, kernel)

    def contour(self):
        contours, hierarchy = cv2.findContours(self.closed_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        real_contours = []
        self.coordinate_list = []
        for cnt in contours:        
            x,y,w,h = cv2.boundingRect(cnt)
            if (not (w < 20 or h < 20)):                   
                self.coordinate_list.append((x, y))
                real_contours.append(cnt)

        cv2.drawContours(self.resistors_im, real_contours, -1, (0,255,0), 3)

    def detect_resistors(self):

        # 1) APPLY CLAHE --> Helps reduce the effects of reflection off the resistor surface
        self.clahe()

        # 2) FILTER OUT BACKGROUND
        self.filter_background()

        # 3) MORPHOLOGICAL CLOSING --> Fills small holes in image, resistor fragments are connected to form
        #                              a resistor blob that can be outlined
        self.morphological_closing()

        # 4) FIND CONTOURS --> Find parts of image that are connected
        self.contour()
        return self.coordinate_list

    def get_resistor_image(self):
        return self.resistors_im
