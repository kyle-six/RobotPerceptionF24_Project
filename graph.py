import numpy as np


class Node:

    def __init__(self, vlad, id):
        self.vlad = vlad
        self.id = id

    def distance(self, node):
        return np.linalg.norm(self.vlad - node.vlad)

    def textures_in_common(self, node):
        set1 = self.vlad
        set2 = node.vlad
        return len(set1.intersection(set2))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False
