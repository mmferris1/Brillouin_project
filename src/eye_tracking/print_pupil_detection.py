
import cv2
import numpy as np
import math
from time import sleep


class PupilDetection:

    def __init__(self):

        self.kernel_size = (5, 5) #Size of the kernel for morphological operations (close/open)
        self.inpaint_radius = 3 #Radius for inpainting glint artifacts
        self.threshold_offset = 40 #added to the quantile-based threshold for binarizing the pupil
        # originally +5 (probably for with near IR light
        self.blur_ksize = 5 #Kernel size for median blur (smoothing before thresholding)


        self.min_area_frac = 0.25   # Fraction of smallest pupil area
        self.max_area_frac = 8.0    # fraction for large pupils
        #Relative bounds on contour area not too big not too small

        self.circularity_threshold = 1.2  # How circular the contour must be - accepts shapes close to circles (1.0 = perfect circle)
        self.max_extend_threshold = 0.8   # Rejects non-pupil blobs that nearly fill bounding box

        self.ellipse_thickness = 4 #thickness of radius ring
        self.center_dot_radius = 4  #thickness of the center dot


        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        # define pupil detection constants here

    # imageData is a numpy ndarray
    # TODO: clean up constants, like the contour area discriminator, circularity etc

    def DetectPupil(self, imageData, radiusGuess):
        if len(imageData.shape) == 2 or imageData.shape[2] == 1:
            image = cv2.cvtColor(imageData, cv2.COLOR_GRAY2BGR)
        else:
            image = imageData.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, glint_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        gray = cv2.inpaint(gray, glint_mask, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA)
        grayblur = cv2.medianBlur(gray, self.blur_ksize)

        image_area = image.shape[0] * image.shape[1]
        pupil_area = math.pi * radiusGuess ** 2
        pupilFrac = pupil_area / image_area
        threshold_val = np.quantile(grayblur, pupilFrac) + self.threshold_offset
        print(f"[DEBUG] Threshold value: {threshold_val:.2f}")
        _, threshold = cv2.threshold(grayblur, threshold_val, 255, 0)
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contour_result = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = contour_result[-2]

        closed = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        drawing = np.copy(image)
        center = (np.nan, np.nan)
        ellipse = None
        pupil_radius = float('nan')

        for contour in contours:
            contour = cv2.convexHull(contour)
            contour_area = cv2.contourArea(contour)
            if contour_area < self.min_area_frac * pupil_area or contour_area > self.max_area_frac * pupil_area:
                continue
            contour_circumference = cv2.arcLength(contour, True)
            contour_circularity = contour_circumference ** 2 / (4 * math.pi * contour_area)
            if contour_circularity > self.circularity_threshold:
                continue
            bounding_box = cv2.boundingRect(contour)
            contour_extend = contour_area / (bounding_box[2] * bounding_box[3])
            if contour_extend > self.max_extend_threshold:
                continue
            m = cv2.moments(contour)
            if m['m00'] != 0:
                center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                cv2.circle(drawing, center, self.center_dot_radius, (0, 255, 0), -1)
            try:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0), thickness=self.ellipse_thickness)
                pupil_radius = (ellipse[1][0] + ellipse[1][1]) / 4
                break
            except:
                ellipse = None
                pupil_radius = float('nan')

        print("[DEBUG] Returning final 8-tuple")
        return (drawing, center, ellipse, pupil_radius, gray, grayblur, threshold, closed)

