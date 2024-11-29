import networkx as nx
from sklearn.neighbors import BallTree
import os
import cv2
import numpy as np

# Maybe we should consider having 2 thresholds. One for similarity between consecutive images, and one for determining loop closure
threshold = 1.25
graph = nx.Graph()


class Node:
    def __init__(self, vlad, id: int):
        self.vlad = vlad
        self.id: int = id
        # self.images: set[int] = {}

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def is_similar(self, other_vlad) -> bool:
        similarity = np.dot(other_vlad, self.vlad) / (
            np.linalg.norm(other_vlad) * np.linalg.norm(self.vlad)
        )
        # distance = other_vlad - self.vlad
        # magnitude = np.linalg.norm(distance)
        # print("magnitude: ", magnitude)
        return similarity < threshold

    # def add_image_id(self, img: int):
    #     self.images.append(img)


def pre_nav_build_graph(tree: BallTree, database: list, save_dir: str):
    # Initial node
    current_node = Node(database[0], 0)
    current_node.add_image_id(0)
    graph.add_node(current_node)

    files = os.listdir(save_dir)
    for i in range(1, len(files)):

        # Stay on the same node?
        if current_node.is_similar(node.vlad):
            # current_node.add_image_id(i)  # Add new associated image
            continue

        # Close the loop?
        loop_found = False
        for node in graph.nodes():
            if node.is_similar(database[i]):
                graph.add_edge(current_node.id, node.id)
                current_node = node
                loop_found = True
                break

        # New Node
        if not loop_found:
            current_node = Node(database[i], i)
            # current_node.add_image_id(i)
            graph.add_node(current_node)


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
        self.ballTree = BallTree(self.node_vlads)
        if not node.id == 0:
            self.graph.add_edge(self.current_node_id, node.id)
        print(f"New Node: {node.id}")
        return node.id

    def create_loop(self, previous_node_id: int) -> int:
        self.graph.add_edge(self.current_node_id, previous_node_id)
        print(f"loop: {self.current_node_id} -> {previous_node_id}")
        return previous_node_id

    def live_maze_graph(self, vlad):
        # Initial node
        if len(self.graph.nodes()) == 0:
            self.current_node_id = self.add_node(vlad)
            return self.current_node_id

        distances, indeces = self.ballTree.query(
            vlad.reshape(1, -1), min(2, len(self.graph.nodes()))
        )

        # New Node
        if distances[0][0] > threshold:
            self.current_node_id = self.add_node(vlad)

        # Found a loop: Image belongs to current and previous node
        elif (
            len(indeces) > 1
            and indeces[0][0] == self.current_node_id
            and distances[0][1] < threshold
        ):
            self.current_node_id = self.create_loop(indeces[0][1])

        # Found a loop: Image belongs to previous node
        elif indeces[0][0] != self.current_node_id and distances[0][0] < threshold:
            self.current_node_id = self.create_loop(indeces[0][0])

        # Stay on the same node: Image belongs to current node
        elif indeces[0][0] == self.current_node_id and distances[0][0] < threshold:
            pass
        return self.current_node_id
