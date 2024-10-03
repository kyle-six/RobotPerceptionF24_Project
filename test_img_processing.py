import cv2 as cv
import numpy as np
import math

# img = cv.imread("data/images/0.jpg")

# gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# #(H, S, V) = cv.split(img_hsv)

# ret, thresh = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)

# cv.imwrite("clean_threshold.png", thresh)

# dst = cv.Canny(thresh, 0, 255, None, 3)
# cv.imwrite("clean_threshold_canny.png", dst)
    
# # Copy edges to the images that will display the results in BGR
# cdst = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

# lines = cv.HoughLines(dst, 1, np.pi / 180, 100, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

# cv.imshow("Source", thresh)
# cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

# cv.waitKey()

# function for drawing realtime over given frame
def detect_draw_lines(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    ret, binary = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)

    edges = cv.Canny(binary, 0, 255, None, 3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 110, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(frame, pt1, pt2, (0,0,255), 1, cv.LINE_AA)