import networkx as nx
from sklearn.neighbors import BallTree
import os
import cv2
import numpy as np

# Maybe we should consider having 2 thresholds. One for similarity between consecutive images, and one for determining loop closure
threshold = ...
graph = nx.Graph() 

class Node:
    def __init__(self, vlad, id: int):
        self.vlad                = vlad
        self.id : int            = id
        self.images : set[int]  = {}

    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False
    
    def is_similar(self, other_vlad) -> bool:
        distance = other_vlad - self.vlad
        magnitude = np.linalg.norm(distance)
        return magnitude < threshold
    
    def add_image_id(self, img: int):
        self.images.append(int)

def pre_nav_build_graph(tree: BallTree, database: list, save_dir: str):
    # Initial node
    current_node = Node(database[0], 0)
    current_node.add_image_id(0)
    graph.add_node(current_node)

    files = os.listdir(save_dir)
    for i in range(1, len(files)):
        # Stay on the same node? 
        if current_node.is_similar(node.vlad):
            current_node.add_image_id(i) # Add new associated image
            continue

        # Close the loop?
        loop_found = False
        for node in graph.nodes():
            if node.is_similar(current_node.vlad):
                graph.add_edge(current_node.id, node.id)
                current_node = node.id
                loop_found = True
                break
        
        # New Node
        if not loop_found:
            current_node = Node(database[i], i)
            current_node.add_image_id(i)
            graph.add_node(current_node)
            