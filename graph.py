import numpy as np


class Node:

    def __init__(self, vlad, id):
        self.vlad = vlad
        self.id = id

    def distance(self, node):
        return np.linalg.norm(self.vlad - node.vlad)

    def similarity(self, node):
        set1 = self.vlad
        set2 = node.vlad
        intersection_size = len(set1.intersection(set2))
        union_size = len(set1.union(set2))
        if union_size == 0 or intersection_size == 0:
            return 0

        similarity = (intersection_size / union_size) * 100
        return similarity

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False
