import pickle
import networkx as nx
from sklearn.neighbors import BallTree
import os
import cv2
import numpy as np

# Maybe we should consider having 2 thresholds. One for similarity between consecutive images, and one for determining loop closure
threshold = 1.25


codebook = pickle.load(open("codebook.pkl", "rb"))
sift = cv2.SIFT_create()


def get_VLAD(img):
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
    _, des = sift.detectAndCompute(img, None)
    # We then predict the cluster labels using the pre-trained codebook
    # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
    pred_labels = codebook.predict(des)
    # Get number of clusters that each descriptor belongs to
    centroids = codebook.cluster_centers_
    # Get the number of clusters from the codebook
    k = codebook.n_clusters
    VLAD_feature = np.zeros([k, des.shape[1]])

    # Loop over the clusters
    for i in range(k):
        # If the current cluster label matches the predicted one
        if np.sum(pred_labels == i) > 0:
            # Then, sum the residual vectors (difference between descriptors and cluster centroids)
            # for all the descriptors assigned to that clusters
            # axis=0 indicates summing along the rows (each row represents a descriptor)
            # This way we compute the VLAD vector for the current cluster i
            # This operation captures not only the presence of features but also their spatial distribution within the image
            VLAD_feature[i] = np.sum(des[pred_labels == i, :] - centroids[i], axis=0)
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


class Node:
    def __init__(self, vlad, id: int):
        self.vlad = vlad
        self.id: int = id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False

    # def is_similar(self, other_vlad) -> bool:
    #     similarity = np.dot(other_vlad, self.vlad) / (
    #         np.linalg.norm(other_vlad) * np.linalg.norm(self.vlad)
    #     )
    #     return similarity < threshold


class MazeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.current_node_id = None
        self.ballTree = None
        self.node_vlads = []
        self.number_nodes = 0

    def add_node(self, vlad) -> int:
        node = Node(vlad, self.number_nodes)
        self.number_nodes += 1
        self.graph.add_node(node)
        self.node_vlads.append(node.vlad)
        if self.number_nodes < 10 or self.number_nodes % 10 == 0:
            self.ballTree = BallTree(self.node_vlads)
        if not node.id == 0:
            self.graph.add_edge(self.current_node_id, node.id)
        print(f"New Node: {node.id}")
        return node.id

    def create_loop(self, previous_node_id: int) -> int:
        self.graph.add_edge(self.current_node_id, previous_node_id)
        print(f"loop: {self.current_node_id} -> {previous_node_id}")
        return previous_node_id

    def create_graph(self):
        files = os.listdir("data/images/")
        for ix in range(1, int(len(files) / 4 - 1)):
            i = ix * 4
            print(i)
            path = "data/images/" + str(i) + ".jpg"
            img = cv2.imread(path)
            self.add_frame(get_VLAD(img))
        pickle.dump(self.graph, open("big_graph.pickle", "wb"))
        pickle.dump(self.ballTree, open("node_ballTree.pickle", "wb"))

    def add_frame(self, vlad):
        # Initial node
        if len(self.graph.nodes()) == 0:
            self.current_node_id = self.add_node(vlad)
            return self.current_node_id

        distances, indices = self.ballTree.query(
            vlad.reshape(1, -1), min(2, self.number_nodes)
        )

        # New Node
        if distances[0][0] > threshold:
            self.current_node_id = self.add_node(vlad)

        # Found a loop: Image belongs to current and old node
        elif (
            len(indices) > 1
            and indices[0][0] == self.current_node_id
            and distances[0][1] < threshold
        ):
            self.current_node_id = self.create_loop(indices[0][1])

        # Found a loop: Image belongs to old node
        elif indices[0][0] != self.current_node_id and distances[0][0] < threshold:
            self.current_node_id = self.create_loop(indices[0][0])

        # Stay on the same node: Image belongs to current node
        elif indices[0][0] == self.current_node_id and distances[0][0] < threshold:
            pass
        return self.current_node_id


# m = MazeGraph()
# m.create_graph()
big_graph = pickle.load(open("big_graph.pickle", "rb"))

from matplotlib import pylab
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pyg


def save_graph(graph, file_name):
    # initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis("off")
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=20)
    nx.draw_networkx_edges(graph, pos)
    # nx.draw_networkx_labels(graph, pos)

    # cut = 1.00
    # xmax = cut * max(xx for xx, yy in pos.values())
    # ymax = cut * max(yy for xx, yy in pos.values())
    # plt.xlim(0, xmax)
    # plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig


nx.drawing.nx_pydot.write_dot(big_graph, "graph_dot.dot")
# nx.write_dot(big_graph, "graph_dot.dot")
# save_graph(big_graph, "my_graph.pdf")

# fig = plt.figure(1)
# nx.draw(big_graph)
# plt.savefig("big_graph2.pdf", bbox_inches="tight")
# it can also be saved in .svg, .png. or .ps formats
