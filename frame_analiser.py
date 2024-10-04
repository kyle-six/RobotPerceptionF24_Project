import numpy as np
import cv2
from texture_classifier import TextureClassifier


class PlotUtils:
    def plot_lines(self, lines, img, write=True):
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
        if write:
            cv2.imwrite("data/out/linesDetected.jpg", img)

    def plot_rectangles(self, rectangles, img, write=False):
        for rect in rectangles:
            for i in range(len(rect) - 1):
                start = tuple(rect[i])
                end = tuple(rect[i + 1])
                cv2.line(img, start, end, (0, 0, 255), 2)

            start = tuple(rect[-1])
            end = tuple(rect[0])
            cv2.line(img, start, end, (0, 0, 255), 2)
        if write:
            cv2.imwrite("data/out/rectanglesDetected.jpg", img)


class FrameAnaliser:
    def __init__(self):
        self.plotting = PlotUtils()
        self.texture_classifier = TextureClassifier()

    def perspective_transform(self, image, rect):
        sum_of_points = rect.sum(axis=1)
        top_left = rect[np.argmin(sum_of_points)]
        bottom_right = rect[np.argmax(sum_of_points)]
        diff_of_points = np.diff(rect, axis=1)
        top_right = rect[np.argmin(diff_of_points)]
        bottom_left = rect[np.argmax(diff_of_points)]
        rect_n = (top_left, bottom_left, bottom_right, top_right)
        width = max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
        height = max(
            np.linalg.norm(bottom_right - top_right),
            np.linalg.norm(bottom_left - top_left),
        )
        dst_points = np.array(
            [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(
            np.array(rect_n, dtype="float32"), dst_points
        )
        transformed = cv2.warpPerspective(image, matrix, (int(width), int(height)))
        return transformed

    def white_wall_black_floor(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        return binary

    def find_vertical_lines(self, binary):
        canny = cv2.Canny(binary, 50, 200, None, 3)
        lines_ver = cv2.HoughLines(
            canny, 1, np.pi / 180, 65, min_theta=np.pi, max_theta=np.pi
        )
        return lines_ver

    def seperate_white_walls(self, binary):
        lines_ver = self.find_vertical_lines(binary)
        if not lines_ver is None:
            self.plotting.plot_lines(lines_ver, binary)
        return binary

    def find_big_rectangles(self, binary):
        contours, hierarchy = cv2.findContours(binary.astype(np.uint8), 1, 2)
        big_rectangles = []
        for cnt in contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if abs(approx[0][0][1] - approx[1][0][1]) > 20 and len(approx) == 4:
                big_rectangles.append(np.array([(a[0][0], a[0][1]) for a in approx]))
        return big_rectangles

    def extract_wall_images(self, enclosing_rectangles, img):
        wall_images = []
        for enc_rect in enclosing_rectangles:
            wall_images.append(self.perspective_transform(img, enc_rect))
        return wall_images

    def find_textures_in_image(self, img):

        binary = self.white_wall_black_floor(img)
        seperated_walls_binary = self.seperate_white_walls(binary)
        rectangles = self.find_big_rectangles(seperated_walls_binary)
        wall_images = self.extract_wall_images(rectangles, img)
        self.plotting.plot_rectangles(rectangles, img, write=True)
        textures = set()
        for wall_img in wall_images:
            textures = textures.union(
                self.texture_classifier.find_textures_on_wall(wall_img, plot=True)
            )
        return textures


frame_analiser = FrameAnaliser()
img = cv2.imread("data/images/90.jpg")
print(frame_analiser.find_textures_in_image(img))
