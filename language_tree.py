class Tree:
    def __init__(self, data, left=None, right=None):
        self.left = left 
        self.right = right
        self.data = data

    def __str__(self):
        return "Node value is %s" % (self.data)

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

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.data
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.data
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.data
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.data
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


def distance_matrix(nodes, tiny_dmatrix, dgrms):
    """
    Computes distance matrix with complete linkage 
    """
    d_matrix = np.full((len(nodes), len(nodes)), np.nan)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if isinstance(nodes[i], Tree) and not isinstance(nodes[j], Tree):
                nlist = Tree.node_list(nodes[i])
                ids = [node.data for node in nlist]
                d_matrix[i,j] = max([[tiny_dmatrix[i,j] for i in ids] for j in ids])
            elif isinstance(nodes[j], Tree) and not isinstance(nodes[i], Tree): 
                nlist = Tree.node_list(nodes[j])
                ids = [node.data for node in nlist]
                d_matrix[i,j] = max([[tiny_dmatrix[i,j] for i in ids] for j in ids])
            elif isinstance(nodes[i], Tree) and isinstance(nodes[j], Tree):
                nlist_1 = Tree.node_list(nodes[i])
                nlist_2 = Tree.node_list(nodes[j])
                ids_1 = [node.data for node in nlist_1]
                ids_2 = [node.data for node in nlist_2]
                d_matrix[i,j] = max([[tiny_dmatrix[i,j] for i in ids_1] for j in ids_2])
            else: 
                d_matrix[i,j] = bottleneck(dgrms[nodes[i].data], dgrms[nodes[j].data])
            d_matrix[j,i] = d_matrix[i,j]
        return d_matrix
    
def hclustering(dgrms, homology=0):
    nodes = [Tree(i) for i in range(len(dgrms))]
    new_dgrms = [dgrms[i][homology] for i in dgrms]
    tiny_dmatrix = np.full((len(nodes), len(nodes)), np.nan)                
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            tiny_dmatrix[i,j] = bottleneck(new_dgrms[nodes[i].data], new_dgrms[nodes[j].data])
    while len(nodes) > 1:
        print(nodes)
        d_matrix = distance_matrix(nodes, tiny_dmatrix, new_dgrms)
        i, j = np.unravel_index(np.argmin(d_matrix), d_matrix.shape)
        node = Tree(None, left=nodes.pop(i), right=nodes.pop(j))
        nodes.append(node)        
    return nodes[0]
    
