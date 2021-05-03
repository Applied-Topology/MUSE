import numpy as np
import matplotlib.pyplot as plt

from persim import sliced_wasserstein, bottleneck


class Tree:
    def __init__(self, data, left=None, right=None):
        self.left = left
        self.right = right
        self.data = data
    def __str__(self):
        return "Data: %s" % self.data

    @staticmethod
    def node_list(root):
        nlist = []
        queue = [root]
        while len(queue) > 0:
            node = queue.pop()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            else:
                nlist.append(node)
        return nlist

    def display(self, keys):
        lines, *_ = self._display_aux(keys)
        for line in lines:
            print(line)

    def _display_aux(self, keys):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = keys[self.data]
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux(keys)
            s = keys[self.data]
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux(keys)
            s = keys[self.data]
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux(keys)
        right, m, q, y = self.right._display_aux(keys)
        s = keys[self.data]
        u = len(s)+1
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


def distance_matrix(nodes, tiny_dmatrix, linkage = "complete"):
    """
    Computes distance matrix with complete linkage
    """
    d_matrix = np.full((len(nodes), len(nodes)), np.nan)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            nlist_1 = Tree.node_list(nodes[i])
            nlist_2 = Tree.node_list(nodes[j])
            ids_1 = [node.data for node in nlist_1]
            ids_2 = [node.data for node in nlist_2]
            if linkage == "complete":
                d_matrix[i,j] = max([tiny_dmatrix[i,j] for i in ids_1 for j in ids_2])
            else:
                d_matrix[i,j] = min([tiny_dmatrix[i,j] for i in ids_1 for j in ids_2])
#             if isinstance(nodes[i], Tree) and not isinstance(nodes[j], Tree):
#                 nlist = Tree.node_list(nodes[i])
#                 ids = [node.data for node in nlist]
#                 d_matrix[i,j] = max([tiny_dmatrix[i,j] for i in ids])
# #                 d_matrix[i,j] = min([tiny_dmatrix[i,j] for i in ids for j in ids])
#             elif isinstance(nodes[j], Tree) and not isinstance(nodes[i], Tree):
#                 nlist = Tree.node_list(nodes[j])
#                 ids = [node.data for node in nlist]
# #                 d_matrix[i,j] = max([tiny_dmatrix[i,j] for i in ids for j in ids])
#                 d_matrix[i,j] = min([tiny_dmatrix[i,j] for i in ids for j in ids])
#             elif isinstance(nodes[i], Tree) and isinstance(nodes[j], Tree):

#             else:
#                 d_matrix[i,j] = tiny_dmatrix[nodes[i].data, nodes[j].data]
    return d_matrix


def hclustering(dgrms, homology=1, dist='sw'):
    nodes = [Tree(i) for i in range(len(dgrms))]
    new_dgrms = [dgrms[i][homology] for i in dgrms]
    tiny_dmatrix = np.full((len(nodes), len(nodes)), np.nan)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if dist == 'sw':
                tiny_dmatrix[i,j] = sliced_wasserstein(new_dgrms[nodes[i].data], new_dgrms[nodes[j].data])
            else:
                tiny_dmatrix[i,j] = bottleneck(new_dgrms[nodes[i].data], new_dgrms[nodes[j].data])
            tiny_dmatrix[j,i] = tiny_dmatrix[i,j]
#     langs = list(dgrms.keys())
#     langs.append('')
    while len(nodes) > 1:
#         a =[print(node) for node in nodes]
#         a = [node.display(langs) for node in nodes]
        d_matrix = distance_matrix(nodes, tiny_dmatrix)
#         print(d_matrix)
        i, j = np.unravel_index(np.nanargmin(d_matrix), d_matrix.shape)
        print("The minimum is ", d_matrix[i,j])
        node = Tree(-1, left=nodes[i], right=nodes[j])
        nodes = [nodes[k] for k in range(len(nodes)) if k not in [i,j]]
        nodes.append(node)
    return nodes[0], tiny_dmatrix


def hclustering_pass_dmatrix(dgrms, tiny_dmatrix, linkage = "complete"):
    ''' Pass distance matrix, can paralelize pairwise distance computation outside this function
    '''
    nodes = [Tree(i) for i in range(len(dgrms))]
    while len(nodes) > 1:
        d_matrix = distance_matrix(nodes, tiny_dmatrix, linkage)
        i, j = np.unravel_index(np.nanargmin(d_matrix), d_matrix.shape)
        print("The minimum is ", d_matrix[i,j])
        node = Tree(-1, left=nodes[i], right=nodes[j])
        nodes = [nodes[k] for k in range(len(nodes)) if k not in [i,j]]
        nodes.append(node)
    return nodes[0]



def hclustering_all(dgrms, homology=1, dist='sw'):
    nodes = [Tree(i) for i in range(len(dgrms))]
#     new_dgrms = [dgrms[i][homology] for i in dgrms]
    new_dgrms = [np.vstack((dgrms[i][0], dgrms[i][1])) for i in dgrms]
    tiny_dmatrix = np.full((len(nodes), len(nodes)), np.nan)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if dist == 'sw':
                tiny_dmatrix[i,j] = sliced_wasserstein(new_dgrms[nodes[i].data], new_dgrms[nodes[j].data])
            else:
                tiny_dmatrix[i,j] = bottleneck(new_dgrms[nodes[i].data], new_dgrms[nodes[j].data])
            tiny_dmatrix[j,i] = tiny_dmatrix[i,j]
    langs = list(dgrms.keys())
    langs.append('')
    while len(nodes) > 1:
#         a =[print(node) for node in nodes]
        a = [node.display(langs) for node in nodes]
        d_matrix = distance_matrix(nodes, tiny_dmatrix, new_dgrms)
        i, j = np.unravel_index(np.nanargmin(d_matrix), d_matrix.shape)
        print("The minimum is ", d_matrix[i,j])
        node = Tree(None, left=nodes[i], right=nodes[j])
        nodes = [nodes[k] for k in range(len(nodes)) if k not in [i,j]]
        nodes.append(node)
    return nodes[0], tiny_dmatrix

def make_latex_tree(tree,langs):
    print(chr(92)+'begin{forest}\n[',end='')
    _make_latex_tree(tree,langs)
    print(']\n'+ chr(92) +'end{forest}',end='')

def _make_latex_tree(tree, langs):
    if not tree.left and not tree.right:
        print(langs[tree.data], end='')
    elif tree.data:
        print('|[', end='')
        _make_latex_tree(tree.left,langs)
        print(']',end='')
        print('[',end='')
        _make_latex_tree(tree.right,langs)
        print(']',end='')
