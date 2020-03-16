import cv2
import matplotlib.pyplot as plt
import numpy as np
from ScanImg import Scan

def fity(img, start_percent, stop_percent, threshold,str):
    img_width, img_height = img.shape[::-1]
    best_location_count = -1
    best_locations = []
    best_scale = 1

    SAMx=50
    SAMy=50

    #plt.axis([0, 2, 0, 1])
    #plt.show(block=False)

    x = []
    y = []
    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 4)]:          #MAKING CHANGE CUZ TLE 3=>6
        locations = []
        location_count = 0
        fx=int(scale*SAMx)
        fy=int(scale*SAMy)

        result = Scan(str,img,fx,fy)
        result = np.where(result >= threshold)
        location_count += len(result[0])
        locations += [result]

        print("scale: {0}, hits: {1}".format(scale, location_count))
        #x.append(location_count)
        #y.append(scale)
        #plt.plot(y, x)
        #plt.pause(0.00001)
        if (location_count > best_location_count):
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
            plt.axis([0, 2, 0, best_location_count])
        elif (location_count < best_location_count):
            pass
    #plt.close()

    return best_locations, best_scale
