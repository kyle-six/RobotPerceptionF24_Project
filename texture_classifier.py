import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
import cv2
import pickle


class TextureClassifier:

    def __init__(self, img_folder="data/textures", nr_bins=6, from_scratch=False):
        self.nr_bins = nr_bins
        balltree_path = Path("colored_texture_balltree.pkl")
        if not balltree_path.is_file() or from_scratch:
            self.tree = self.init_balltree(img_folder)
        else:
            with open("colored_texture_balltree.pkl", "rb") as f:
                self.tree = pickle.load(f)

    def color_histogram_features(self, img):

        hist_r = cv2.calcHist([img], [0], None, [self.nr_bins], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [self.nr_bins], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [self.nr_bins], [0, 256])

        # Normalize histograms and flatten them into a single feature vector
        hist_r = hist_r / (np.sum(hist_r) + 1e-6)
        hist_g = hist_g / (np.sum(hist_g) + 1e-6)
        hist_b = hist_b / (np.sum(hist_b) + 1e-6)

        return np.append(
            np.hstack([hist_r, hist_g, hist_b]).flatten(),
            [np.median(img[:, :, 0]), np.median(img[:, :, 1]), np.median(img[:, :, 2])],
        ).reshape(1, -1)

    def init_balltree(self, img_folder):
        img_names = [img for img in os.listdir(img_folder) if img.endswith(".png")]
        img_names.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        feature_vectors = np.zeros((200, self.nr_bins * 3 + 3))
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path).convert("RGB")  # Keep the image in RGB format
            img = np.array(img)  # Convert to numpy array
            color_hist_feats = self.color_histogram_features(img)
            feature_vectors[i] = color_hist_feats
        feature_vectors = np.array(feature_vectors)
        tree = BallTree(feature_vectors, leaf_size=40)
        tree_filename = "colored_texture_balltree.pkl"
        with open(tree_filename, "wb") as f:
            pickle.dump(tree, f)
        return tree

    def find_textures_on_wall(
        self, img, plot=False, nRows=1, mCols=4, tolerance_for_match=5
    ):
        img = np.array(img)  # Convert to numpy array
        sizeX = img.shape[1]
        sizeY = img.shape[0]
        found = set()
        for i in range(0, nRows):
            for j in range(0, mCols):
                roi = img[
                    int(i * sizeY / nRows) : int(i * sizeY / nRows + sizeY / nRows),
                    int(j * sizeX / mCols) : int(j * sizeX / mCols + sizeX / mCols),
                ]
                features = self.color_histogram_features(roi)
                distances, indices = self.tree.query(features, k=1)
                if distances[0][0] <= tolerance_for_match:
                    found.add(indices[0][0])
                if plot:
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
        if plot:
            cv2.imwrite(f"data/out/detected.jpg", img)
        return found
