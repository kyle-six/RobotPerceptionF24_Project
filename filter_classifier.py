import os
import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
import cv2
import pickle


# Color histogram feature extraction
def color_histogram_features(img, bins=32):
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])

    # Normalize histograms and flatten them into a single feature vector
    hist_r = hist_r / (np.sum(hist_r) + 1e-6)
    hist_g = hist_g / (np.sum(hist_g) + 1e-6)
    hist_b = hist_b / (np.sum(hist_b) + 1e-6)

    return np.append(
        np.hstack([hist_r, hist_g, hist_b]).flatten(),
        [np.median(img[:, :, 0]), np.median(img[:, :, 1]), np.median(img[:, :, 2])],
    )


# Load images and compute features
def compute_features_from_colored_images(img_folder, image=None):
    img_names = [1]
    if image is None:
        img_names = [img for img in os.listdir(img_folder) if img.endswith(".png")]
        img_names.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    feature_vectors = []
    for img_name in img_names:

        if not image is None:
            img = image
        else:
            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path).convert("RGB")  # Keep the image in RGB format

        img = np.array(img)  # Convert to numpy array

        # Color histogram features from color image
        color_hist_feats = color_histogram_features(img)

        feature_vectors.append(color_hist_feats)

    return np.array(feature_vectors)


# Main processing pipeline
img_folder = "data/textures"

# Step 1: Compute the feature vectors for all images
print("Extracting features from colored images...")
feature_vectors = compute_features_from_colored_images(img_folder)
print(f"Extracted feature vectors shape: {feature_vectors.shape}")

# Step 2: Build the BallTree for k-NN search
print("Building BallTree...")
tree = BallTree(feature_vectors, leaf_size=40)

# Step 3: Save the BallTree for later use
tree_filename = "colored_texture_balltree.pkl"
with open(tree_filename, "wb") as f:
    pickle.dump(tree, f)

print(f"BallTree saved as {tree_filename}")

# Optional: Save the feature vectors and corresponding image names for later reference
features_filename = "colored_texture_features.npy"
np.save(features_filename, feature_vectors)
print(f"Feature vectors saved as {features_filename}")
