import cv2
import numpy as np

class LaserFocusVisualizer:
    def __init__(self):
        self.laser_lookup = {
            700: (1425, 994),
            2000: (1406, 992),
            3000: (1386, 993),
            5000: (1346, 993),
            6000: (1327, 994),
            7000: (1396, 994),
            8999: (1270, 995),
            9999: (1260, 995),
            10999: (1239, 996),
            11999: (1228, 996),
            12999: (1215, 997),
            13999: (1196, 997),
            14999: (1185, 997),
            15999: (1169, 998),
            16999: (1145, 998),
            17999: (1131, 998),
            18999: (1117, 998),
            19999: (1104, 997),
            20999: (1089, 998),
            21999: (1077, 998),
            22999: (1066, 999),
        }

    def draw_laser_marker(self, image, distance, color=(0, 0, 255), cross_size=5):
        """
        Draw a red cross at the interpolated laser focus point based on distance of the laser.
        """
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        drawing = image.copy()
        distances = sorted(self.laser_lookup.keys())

        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i + 1]
            if d1 <= distance <= d2:
                x1, y1 = self.laser_lookup[d1]
                x2, y2 = self.laser_lookup[d2]
                ratio = (distance - d1) / (d2 - d1)
                x_interp = int(x1 + ratio * (x2 - x1))
                y_interp = int(y1 + ratio * (y2 - y1))

                cv2.drawMarker(drawing, (x_interp, y_interp), color,
                               markerType=cv2.MARKER_CROSS,
                               markerSize=3 * cross_size,
                               thickness=2)
                break

        return drawing

    def get_laser_pixel(self, distance):
        distances= sorted(self.laser_lookup.keys())

        for i in range(len(distances) - 1):
            d1,d2 = distances[i], distances[i+1]
            if d1 <= distance <=2:
                x1,y1 = self.laser_lookup[d1]
                x2,y2 = self.laser_lookup[d2]
                ratio = (distance - d1) / (d2 -d1)
                x_interp = int(x1 + ratio * (x2 - x1))
                y_interp = int(y1 + ratio * (y2 - y1))
                return (x_interp, y_interp)

        return(np.nan, np.nan)
