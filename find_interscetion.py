import cv2
import numpy as np


def get_lines():
    img = cv2.imread("data/images/39.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(binary, 50, 200, None, 3)
    cv2.imwrite("binary.jpg", canny)
    lines_hor = cv2.HoughLines(canny, 1, np.pi / 180, 85, None, 0, 0)
    lines_ver = cv2.HoughLines(
        canny, 1, np.pi / 180, 60, min_theta=np.pi, max_theta=np.pi
    )
    if (
        lines_hor is None
        or lines_ver is None
        or len(lines_hor) < 1
        or len(lines_ver) < 1
    ):
        return 0
    lines = np.vstack((lines_hor, lines_ver))

    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite("linesDetected.jpg", img)


if __name__ == "__main__":
    get_lines()
