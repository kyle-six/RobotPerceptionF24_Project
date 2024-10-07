import datetime
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import cv2
import pickle


class TextureClassifier:

    def __init__(self, img_folder="data/textures", nr_bins=6, from_scratch=False):
        self.nr_bins = nr_bins
        balltree_path = Path("vlad_tree.pkl")
        self.gabor_kernels = self.create_gabor_kernels()
        self.sift = cv2.SIFT_create()
        # if not balltree_path.is_file() or from_scratch:
        #     # self.tree = self.init_balltree(img_folder)
        #     self.vlad_tree = self.init_sift_balltree(img_folder)
        # else:
        #     with open("colored_texture_balltree.pkl", "rb") as f:
        #         self.tree = pickle.load(f)

        if not balltree_path.is_file() or from_scratch:
            # self.tree = self.init_balltree(img_folder)
            self.vlad_tree = self.init_sift_balltree(img_folder)
        else:
            with open("vlad_tree.pkl", "rb") as f:
                self.vlad_tree = pickle.load(f)
            with open("codebook_tex.pkl", "rb") as f:
                self.codebook = pickle.load(f)

    def compute_sift_features(self, img):
        """
        Compute SIFT features for images in the data directory
        """
        _, des = self.sift.detectAndCompute(img, None)
        return des
        # length = len(os.listdir("data/textures/"))
        # print(length)
        # sift_descriptors = list()
        # for i in range(length):
        #     path = str(i) + ".png"
        #     img = cv2.imread(os.path.join(self.save_dir, path))
        #     # Pass the image to sift detector and get keypoints + descriptions
        #     # We only need the descriptors
        #     # These descriptors represent local features extracted from the image.
        #     _, des = self.sift.detectAndCompute(img, None)
        #     # Extend the sift_descriptors list with descriptors of the current image
        #     sift_descriptors.extend(des)
        # return np.asarray(sift_descriptors)

    def create_gabor_kernels(self):
        kernels = []
        ksize = 11  # Size of the filter
        for theta in np.arange(0, np.pi, np.pi / 4):  # Loop through orientations
            for sigma in [1, 3]:  # Different frequencies
                for lamda in np.arange(0, np.pi, np.pi / 2):  # Wavelengths
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F
                    )
                    kernels.append(kernel)
        return kernels

    def gabor_features(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            mean, var = np.mean(filtered), np.var(filtered)
            features.extend([mean, var])
        return np.array(features).reshape(1, -1)

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
        text_sift_features = np.zeros((200, self.nr_bins * 3 + 3 + 32))
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(img_folder, img_name)
            img = cv2.imread(img_path)
            img = np.array(img)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            features = self.compute_sift_features(img)
            feature_vectors[i] = features
        feature_vectors = np.array(feature_vectors)
        tree = BallTree(feature_vectors, leaf_size=5)
        tree_filename = "colored_texture_balltree.pkl"
        with open(tree_filename, "wb") as f:
            pickle.dump(tree, f)
        return tree

    def init_sift_balltree(self, img_folder):
        img_names = [img for img in os.listdir(img_folder) if img.endswith(".png")]
        img_names.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        if not Path("codebook_tex.pkl").is_file():
            feature_vectors = []

            for i, img_name in enumerate(img_names):
                img_path = os.path.join(img_folder, img_name)
                img = cv2.imread(img_path)
                img = np.array(img)
                sift_features = self.compute_sift_features(img)
                if not sift_features is None:
                    feature_vectors.extend(sift_features)
                else:
                    print(i)
            feature_vectors = np.array(feature_vectors)
            self.codebook = KMeans(
                n_clusters=64, init="k-means++", n_init=10, verbose=1
            ).fit(feature_vectors)
            pickle.dump(self.codebook, open("codebook_tex.pkl", "wb"))
        else:
            with open("codebook_tex.pkl", "rb") as f:
                self.codebook = pickle.load(f)
        self.database = []
        for i, img_name in enumerate(img_names):
            if not (i == 17 or i == 42):
                img_path = os.path.join(img_folder, img_name)
                img = cv2.imread(img_path)
                img = np.array(img)

                VLAD = self.get_VLAD(img)
                self.database.append(VLAD)

        tree = BallTree(self.database, leaf_size=60)
        pickle.dump(tree, open("vlad_tree.pkl", "wb"))
        return tree

    def get_features(self, img):
        hist_feature = self.color_histogram_features(img)
        gab_feature = self.gabor_features(img)
        return np.hstack((hist_feature, gab_feature))

    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # Get the number of clusters from the codebook

        if des is None:
            return None

        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(
                    des[pred_labels == i, :] - centroids[i], axis=0
                )
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def find_textures_on_wall_vlad(
        self, img, plot=False, nRows=1, mCols=1, tolerance_for_match=40
    ):
        img = np.array(img)  # Convert to numpy array
        found = set()

        VLAD = self.get_VLAD(img)
        if VLAD is None:
            return found
        q_VLAD = VLAD.reshape(1, -1)

        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        # The '1' indicates the we want 1 nearest neighbor
        d, indices = self.vlad_tree.query(q_VLAD, 3)
        # if distances[0][0] <= tolerance_for_match:
        pattern = indices[0][0]
        for i, pattern in enumerate(indices[0]):
            if d[0][i] < 1.25:
                if pattern > 16:
                    pattern += 1  # cause 17 doesnt work
                if pattern > 41:
                    pattern += 1  # cause 42 doesnt work
                pattern += 1  # for 0 indexing
                found.add(pattern)
        if plot:
            cv2.imwrite(
                "data/out/detected.jpg",
                img,
            )

        return found

    def find_textures_on_wall(
        self, img, plot=False, nRows=1, mCols=4, tolerance_for_match=40
    ):
        img = cv2.GaussianBlur(img, (5, 5), 0)
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
                features = self.get_features(roi)
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
            cv2.imwrite("data/out/detected.jpg", img)
        return found


# TextureClassifier()
