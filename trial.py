import pickle
import numpy as np
import cv2
from filter_classifier import compute_features_from_colored_images


def perspective_transform(image, rect):

    sum_of_points = rect.sum(axis=1)
    top_left = rect[np.argmin(sum_of_points)]
    bottom_right = rect[np.argmax(sum_of_points)]
    diff_of_points = np.diff(rect, axis=1)
    top_right = rect[np.argmin(diff_of_points)]
    bottom_left = rect[np.argmax(diff_of_points)]
    rect_n = (top_left, bottom_left, bottom_right, top_right)
    width = max(
        np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)
    )
    height = max(
        np.linalg.norm(bottom_right - top_right), np.linalg.norm(bottom_left - top_left)
    )
    dst_points = np.array(
        [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(np.array(rect_n, dtype="float32"), dst_points)
    transformed = cv2.warpPerspective(image, matrix, (int(width), int(height)))

    return transformed


def plot_lines(lines, img):
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
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.imwrite("data/out/linesDetected.jpg", img)


def get_lines():
    img = cv2.imread("data/images/43.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    canny = cv2.Canny(binary, 50, 200, None, 3)
    lines_ver = cv2.HoughLines(
        canny, 1, np.pi / 180, 65, min_theta=np.pi, max_theta=np.pi
    )
    cv2.imwrite("data/out/canny.jpg", canny)
    if not lines_ver is None:
        plot_lines(lines_ver, binary)
        thresh = binary
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 2)
        cnt = contours[1]
        for i, cnt in enumerate(contours):

            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if (
                abs(approx[0][0][1] - approx[1][0][1]) > 20 and len(approx) == 4
            ):  # delete small ones
                corners = np.array([(a[0][0], a[0][1]) for a in approx])
                wall_image = perspective_transform(img, corners)
                cv2.imwrite(f"data/out/wall{i}.jpg", wall_image)
                compute_features_from_images(wall_image, i)
                for i in range(len(approx) - 1):
                    start = tuple(approx[i][0])
                    end = tuple(approx[i + 1][0])
                    cv2.line(img, start, end, (0, 0, 255), 2)

                start = tuple(approx[-1][0])
                end = tuple(approx[0][0])
                cv2.line(img, start, end, (0, 0, 255), 2)
        cv2.imwrite("data/out/rect.jpg", img)


def compute_features_from_images(img, img_n):
    with open("colored_texture_balltree.pkl", "rb") as f:
        tree = pickle.load(f)
        img = np.array(img)  # Convert to numpy array
        sizeX = img.shape[1]
        sizeY = img.shape[0]

        nRows = 1
        mCols = 3
        found = set()

        for i in range(0, nRows):
            for j in range(0, mCols):
                roi = img[
                    int(i * sizeY / nRows) : int(i * sizeY / nRows + sizeY / nRows),
                    int(j * sizeX / mCols) : int(j * sizeX / mCols + sizeX / mCols),
                ]
                features = compute_features_from_colored_images("asf", roi)
                distances, indices = tree.query(features, k=1)
                if distances[0][0] < 5:
                    found.add(indices[0][0])
                cv2.putText(
                    img,
                    str(indices[0][0]),
                    (
                        int(j * sizeX / mCols),
                        int(i * sizeY / nRows + sizeY / nRows),
                    ),
                    cv2.FONT_ITALIC,
                    0.3,
                    (0, 255, 0),
                    1,
                )

        cv2.imwrite(f"data/out/detected_{img_n}.jpg", img)
        print(found)


get_lines()
