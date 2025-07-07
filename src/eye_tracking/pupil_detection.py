# Run mako_pupil.py to see just the algorithm in action from a live camera

import cv2
import numpy as np
import math
from time import sleep

class PupilDetection:

    def __init__(self):
        #self.para1 = 1
        self.kernel_size = (5, 5)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        # define pupil detection constants here

    # imageData is a numpy ndarray
    # TODO: clean up constants, like the contour area discriminator, circularity etc
    def DetectPupil(self, imageData, radiusGuess):

        #add color to image for colored pupil ring if necessary
        if len(imageData.shape) == 2 or imageData.shape[2] == 1:
            image = cv2.cvtColor(imageData, cv2.COLOR_GRAY2BGR)
        else:
            image = imageData.copy()

        # Convert to grayscale for pupil detection processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ---- Begin: Glint Inpainting ----
        # Create mask of bright spots (glints)
        _, glint_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Inpaint to remove small glints
        gray = cv2.inpaint(gray, glint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # ---- End: Glint Inpainting ----

        grayblur = cv2.medianBlur(gray,5)
        #grayblur = cv2.blur(gray,(5,5))

        """
        THRESHOLD, used to differentiate dark and light colors, FINDS PUPIL
        First number is used for threshold of dark items - the smaller the number, 
        the darker the item must be. Numbers can vary based on level the LED is set 
        too. At first notch on LED controller, Left eye - 25 is best value; Right eye - 23 
        Works best. This difference is due too the right eye not being illuminated as 
        well as the right eye - maybe due too the angle of the LED, or the nose blocking 
        some of the light. When the right eye looks left or right,  the iris is not lit 
        up enough, so the algorithm circles it as it cannot differentiate it from the pupil. 
        This is why is requires a smaller threshold (darker).
        """
        image_area = image.shape[0] * image.shape[1]
        pupil_area = math.pi * radiusGuess ** 2
        pupilFrac = pupil_area / image_area
        # Approx. fraction of image taken up by pupil
        # original --> pupilFrac = math.pi*radiusGuess*radiusGuess/1000./1000.

        # print('np.mean(gray) =', np.mean(grayBlur))
        # print('np.min(gray) =', np.amin(grayBlur))
        # print('np.quantile(grayBlur, pupilFrac) =', np.quantile(grayBlur, pupilFrac))
        # print('np.quantile(grayBlur, 0.5) =', np.quantile(grayBlur, 0.5))
        retval, threshold = cv2.threshold(grayblur, np.quantile(grayblur, pupilFrac)+50, 255, 0)
            #originally +5
        cv2.imshow("threshold", threshold)

        #cleans up threshold image
        # closed = threshold
        # original closed = cv2.erode(cv2.dilate(threshold, self.kernel, iterations=1), self.kernel, iterations=1)
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        #cv2.imshow("closed", closed)

        #threshold, contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contour_result = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contour_result) == 3:
            # Older OpenCV
            _, contours, hierarchy = contour_result
        else:
            # Newer OpenCV
            contours, hierarchy = contour_result

        closed = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
         ###### for recording video

        drawing = np.copy(image)
        center = (np.nan, np.nan)
        #draws contours on video
        # cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)

        # print("[DetectPupil] num contours: %d" % len(contours))

        for contour in contours:

            contour = cv2.convexHull(contour)
            contour_area = cv2.contourArea(contour)

            #SIZE of pupil ranges, can  adjust to larger or smaller values
            #used to get rid of smaller or larger detections
            #lower bound ~3000, works better in dark room - more dialated pupil
            if contour_area < 0.25*math.pi*radiusGuess*radiusGuess or contour_area > 8*math.pi*radiusGuess*radiusGuess:
                continue
                #changed upper bound from 4 to 8 - this is also dependent on how close we are to the camera - so should be more stable with fixed set up

            #focuses on rounder shapes
            contour_circumference = cv2.arcLength(contour, True)
            contour_circularity = contour_circumference**2 / (4*math.pi*contour_area)
            #closer to 1, the more circluar the elipical shapes
            if contour_circularity > 1.2:
                continue

            #print area
            bounding_box = cv2.boundingRect(contour)

            contour_extend = contour_area / (bounding_box[2] * bounding_box[3])

            # reject the contours with big extend
            if contour_extend > 0.8:
                continue

            # calculate countour center and draw a dot there
            m = cv2.moments(contour)
            if m['m00'] != 0:
                dot = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                cv2.circle(drawing, dot, 4, (0, 255, 0), -1)
                center = (int(m['m01'] / m['m00']), int(m['m10'] / m['m00']))

            # fit an ellipse around the contour and draw it into the image
            try:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0), thickness=4)
                pupil_radius = (ellipse[1][0] + ellipse[1][1]) / 4  # avg of major/minor axes, divide by 2 for radius
                return (drawing, center, ellipse, pupil_radius)
            except:
                pass

        return (drawing, center, None, float('nan'))

if __name__ == "__main__":
    testImage = 128 * np.ones((512,512, 3))
    PD = PupilDetection()
    res = PD.DetectPupil(testImage)
    print("Pupil center:", center)
    print("Pupil radius:", pupil_radius)
    print(res)
    print("done")