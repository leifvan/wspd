from collections import namedtuple


class BinaryNode:
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None



def get_split_tree(points):
    pass