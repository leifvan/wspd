import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class BinaryNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_inner(self):
        return not self.is_leaf()

    def __iter__(self):
        return self.depth_iter()

    def depth_iter(self):
        yield self
        if self.left is not None:
            yield from self.left.depth_iter()
        if self.right is not None:
            yield from self.right.depth_iter()


def point_in_row(p, points):
    return any(np.array_equal(p,pt) for pt in points)


def get_point_idx(p, points):
    for i, pp in enumerate(points):
        if np.array_equal(p, pp):
            return i
    return None


def get_set_repr(s, points):
    indices = [i for i, p in enumerate(points) if point_in_row(p, s)]
    indices.sort()
    return "{" + ",".join([f"{i+1}" for i in indices]) + "}"


def get_split_tree(pts):
    # recursively split, until only leaves remain
    node = BinaryNode(data=pts)

    if len(pts) > 1:
        widths = pts.ptp(axis=0)
        max_i = np.argmax(widths)
        split = widths[:, max_i] / 2 + pts[:, max_i].min()
        node.left = get_split_tree(pts[np.ravel(pts[:, max_i] <= split)])
        node.right = get_split_tree(pts[np.ravel(pts[:, max_i] > split)])
        assert 0 < len(node.left.data) < len(pts)
        assert 0 < len(node.right.data) < len(pts)

    return node


def get_smallest_sphere_radius(points):
    max_width = points.ptp(axis=0).max()
    d = points.shape[1]
    return np.sqrt(d) / 2 * max_width


def find_pairs(child_left, child_right, s, all_points, depth=0):
    pairs = []
    children = (child_left, child_right)
    sets = tuple(c.data for c in children)
    r = tuple(get_smallest_sphere_radius(si) for si in sets)
    max_i = np.argmax(r)

    centers = [np.mean(c, axis=0) for c in sets]
    min_dist = (s + 2) * r[max_i]
    dist = np.linalg.norm(centers[0] - centers[1])

    if dist > min_dist:
        pairs += [sets]
        print("-"*depth+f"{get_set_repr(sets[0], all_points)} <- {dist:.2f} -> {get_set_repr(sets[1], all_points)} YES!"
              f" (r={r[max_i]:.2f} => {min_dist:.2f})")
    else:
        # split set with higher radius
        print("-"*depth+f"{get_set_repr(sets[0], all_points)} <- {dist:.2f} -> {get_set_repr(sets[1], all_points)} NO split!"
              f" (r={r[max_i]:.2f} => {min_dist:.2f})")
        pairs += find_pairs(children[max_i].left, children[1 - max_i], s, all_points, depth=depth+1)
        pairs += find_pairs(children[max_i].right, children[1 - max_i], s, all_points, depth=depth+1)

    return pairs


def get_wspd(tree, s):
    # flattened list from find_pairs calls
    return sum((find_pairs(n.left, n.right, s, tree.data) for n in tree if n.is_inner()), [])


def render_split_tree(tree):
    # make networkx graph
    g = nx.Graph()

    def f(s):
        return get_set_repr(s, tree.data)

    edges = [[[f(n.data), f(n.left.data)],[f(n.data), f(n.right.data)]] for n in tree if n.is_inner()]
    edges = sum(edges, [])
    g.add_edges_from(edges)
    nx.draw(g, with_labels=True)
    plt.show()
