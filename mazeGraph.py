import pickle
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Maybe we should consider having 2 thresholds. One for similarity between consecutive images, and one for determining loop closure
threshold = 1.25


sift = cv2.SIFT_create()


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


from pathlib import Path


class MazeGraph:
    def __init__(self, rebuild=False):
        self.data_path = "data/images/"
        self.pickle_path = "data/pickles/"
        self.img_prefix = "image_"
        self.img_extension = ".png"

        self.graph = nx.Graph()
        self.current_node = None  # Current Node, used during creation and navigation
        self.ballTree = None  # Ball tree over Vlads stored in Graph (Index needs to be translated from index of node to node id)
        self.node_vlads = []  # List of Vlads stored in Graph
        self.number_nodes = 0
        self.nodes = []  # List of Node object stored in Graph
        self.created_video = False

        # Backup ys
        self.graph_pickle_path = self.pickle_path + "graph_loop_closure.pickle"
        self.dot_file_path = self.pickle_path + "graph_loop_closure.dot"
        self.node_list_path = self.pickle_path + "node_list.pickle"
        self.node_vlads_list_path = self.pickle_path + "node_vlad_list.pickle"
        self.balltree_pickle_path = self.pickle_path + "graph_balltree.pickle"
        self.path_video_path = self.pickle_path + "path_video.mp4"
        self.codebook_pickle_path = self.pickle_path + "codebook.pkl"

        # Rebuild codebook if needed
        if Path(self.codebook_pickle_path).is_file() and not rebuild:
            self.codebook = pickle.load(open(self.codebook_pickle_path, "rb"))
        else:
            self.compute_codebook()

        # Rebuild all if graph pickle is not available
        if Path(self.graph_pickle_path).is_file() and not rebuild:
            self.graph = pickle.load(open(self.graph_pickle_path, "rb"))
        else:
            self.create_graph()

        # Rebuild node list if only graph pickle is available
        if Path(self.node_list_path).is_file():
            self.nodes = pickle.load(open(self.node_list_path, "rb"))
            self.node_vlads = pickle.load(open(self.node_vlads_list_path, "rb"))
        else:
            self.rebuild_nodelist()

        # Rebuild balltree if it is not available
        if Path(self.balltree_pickle_path).is_file():
            self.ballTree = pickle.load(open(self.balltree_pickle_path, "rb"))
        else:
            self.ballTree = BallTree(self.node_vlads,leaf_size=20)

    def create_path_video(self, path_to_target, fps=5, size=None):
        """
        Create a video from a list of images and save it as an MP4 file.
        """
        # Read the first image to get the size if not provided
        if size is None:
            first_image = cv2.imread(
                f"{self.data_path}{self.img_prefix}{path_to_target[0]}{self.img_extension}"
            )
            height, width, layers = first_image.shape
            size = (width, height)

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
        out = cv2.VideoWriter(self.path_video_path, fourcc, fps, size)

        # Iterate through the image list and write each image to the video
        for image_id in self.path_to_target[1:]:
            img = cv2.imread(
                f"{self.data_path}{self.img_prefix}{image_id}{self.img_extension}"
            )

            # Resize image if it's not the same size as the video frame size
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size)

            out.write(img)  # Write the frame

        # Release the video writer
        out.release()

    def init_navigation(self, target_img) -> None:
        if not self.created_video:
            self.current_node = self.nodes[0]
            self.target_vlad = self.get_VLAD2(target_img)
            _, self.target_node_index = self.ballTree.query(
                self.target_vlad.reshape(1, -1), 1
            )
            self.target_node_id = self.nodes[self.target_node_index[0][0]].id
            print(f"Calculating Shortest Path to node: {self.target_node_id} ")
            self.path_to_target = nx.shortest_path(
                self.graph, self.current_node.id, self.target_node_id
            )
            self.create_path_video(self.path_to_target)
            self.created_video = True

    def add_node(self, vlad, id) -> Node:
        node = Node(vlad, id)
        self.number_nodes += 1
        self.graph.add_node(node)
        self.nodes.append(node)
        self.node_vlads.append(node.vlad)
        if not node.id == 0:
            self.graph.add_edge(self.current_node.id, node.id)
        print(f"New Node: {node.id}")
        return node

    def approve_potential_loop(
        self, id1, id2, folder="data/images/"
    ) -> bool:
        path1 = f"{folder}{self.img_prefix}{id1}{self.img_extension}"
        img1 = cv2.imread(path1)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        path2 = f"{folder}{self.img_prefix}{id2}{self.img_extension}"
        img2 = cv2.imread(path2)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        approved = None

        def on_key(event):
            nonlocal approved
            if event.key.lower() == "y":
                approved = True
                plt.close(fig)
            elif event.key.lower() == "n":
                approved = False
                plt.close(fig)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img1)
        axs[1].imshow(img2)

        axs[0].set_axis_off()
        axs[1].set_axis_off()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=True)

        return approved

    def loop_detection(self) -> None:
        num_neighbors = 10  # Increased from 4
        id_difference_threshold = 10  # Reduced from 25
        self.ballTree = BallTree(self.node_vlads, leaf_size=20)
        # Look for loops
        for node in self.nodes:
            distances, indices = self.ballTree.query(node.vlad.reshape(1, -1), num_neighbors)
            for i in range(1, num_neighbors):
                candidate_id = self.nodes[indices[0][i]].id
                candidate_dist = distances[0][i]
                if abs(candidate_id - node.id) > id_difference_threshold and candidate_dist < threshold and not self.graph.has_edge(candidate_id, node.id):
                    print(f"Evaluating potential loop closure between nodes {node.id} and {candidate_id}")
                    if self.approve_potential_loop(candidate_id, node.id, self.data_path):
                        self.graph.add_edge(candidate_id, node.id)


    def compute_codebook(self) -> None:
        files = os.listdir(self.data_path)
        sift_descriptors = list()
        for ix in range(0, int(len(files) / 4)):
            i = ix * 4
            path = f"{self.data_path}{self.img_prefix}{i}{self.img_extension}"
            print(path)
            img = cv2.imread(path)
            _, des = sift.detectAndCompute(img, None)
            sift_descriptors.extend(des)
        self.codebook = KMeans(
            n_clusters=1024, init="k-means++", n_init=10, verbose=1
        ).fit(sift_descriptors)
        pickle.dump(self.codebook, open("codebook.pkl", "wb"))

    def create_graph(self) -> nx.Graph:
        files = os.listdir(self.data_path)
        for ix in range(0, int(len(files) / 4)):
            i = ix * 4
            path = f"{self.data_path}{self.img_prefix}{i}{self.img_extension}"
            img = cv2.imread(path)
            self.add_frame(self.get_VLAD2(img), i)
        self.loop_detection()
        self.save_all_files()
        print(f"Created graph with {self.number_nodes+1} nodes")
        return self.graph

    def save_all_files(self):
        pickle.dump(self.ballTree, open(self.balltree_pickle_path, "wb"))
        pickle.dump(self.graph, open(self.graph_pickle_path, "wb"))
        pickle.dump(self.nodes, open(self.node_list_path, "wb"))
        pickle.dump(self.node_vlads, open(self.node_vlads_list_path, "wb"))
        nx.drawing.nx_pydot.write_dot(self.graph, self.dot_file_path)
        # outdeg = self.graph.out_degree() TODO
        # to_keep = [n for n in outdeg if outdeg[n] != 1]
        # G.subgraph(to_keep)
        # nx.drawing.nx_pydot.write_dot(self.subgraph, self.)

    def add_frame(self, vlad, id) -> None:
        # Initial node
        if len(self.graph.nodes()) == 0:
            self.current_node = self.add_node(vlad, id)
            return
        distance = np.linalg.norm(self.current_node.vlad - vlad)
        # New Node
        if distance > threshold:
            self.current_node = self.add_node(vlad, id)

    def rebuild_nodelist(self) -> None:
        """This function is weird, because self.graph.nodes()
        somehow gives a mix of integers and Nodes"""  # lol i can tell this was frustrating!
        for thing in self.graph.nodes():
            if isinstance(thing, Node):
                node = thing
                self.nodes.append(node)
                self.node_vlads.append(node.vlad)
        pickle.dump(self.nodes, open(self.node_list_path, "wb"))
        pickle.dump(self.node_vlads, open(self.node_vlads_list_path, "wb"))

    def get_VLAD2(self, img):
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
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the self.codebook
        k = self.codebook.n_clusters
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


# m = MazeGraph()
# path="data/images/image_5475.png"
# img=cv2.imread(path)
# m.init_navigation(img)


# m.init_navigation(cv2.imread("midterm_data/exploration_data/images/image_5042.png"))
# graph = m.create_graph()
# graph = pickle.load(open("graph_loop_closure.pickle", "rb"))

# import graphviz

# # nx.drawing.nx_pydot.write_dot(graph, "graph_loop_closure.dot")
# dot_graph = graphviz.Source.from_file("graph_loop_closure.dot")

# # Render the graph to an SVG string
# svg_string = dot_graph.pipe(format="svg")

# # Save the SVG string to a file
# with open("graph_loop_closure.svg", "w") as f:
#     f.write(svg_string)
