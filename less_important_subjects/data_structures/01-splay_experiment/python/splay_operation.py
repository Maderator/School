#!/usr/bin/env python3

class Node:
    """Node in a binary tree `Tree`"""

    def __init__(self, key, left=None, right=None, parent=None):
        self.key = key
        self.parent = parent
        self.left = left
        self.right = right
        if left is not None: left.parent = self
        if right is not None: right.parent = self

class Tree:
    """A simple binary search tree"""

    def __init__(self, root=None):
        self.root = root

    def rotate(self, node):
        """ Rotate the given `node` up.

        Performs a single rotation of the edge between the given node
        and its parent, choosing left or right rotation appropriately.
        """
        if node.parent is not None:
            if node.parent.left == node:
                if node.right is not None: node.right.parent = node.parent
                node.parent.left = node.right
                node.right = node.parent
            else:
                if node.left is not None: node.left.parent = node.parent
                node.parent.right = node.left
                node.left = node.parent
            if node.parent.parent is not None:
                if node.parent.parent.left == node.parent:
                    node.parent.parent.left = node
                else:
                    node.parent.parent.right = node
            else:
                self.root = node
            node.parent.parent, node.parent = node, node.parent.parent

    def lookup(self, key):
        """Look up the given key in the tree.

        Returns the node with the requested key or `None`.
        """
        # TODO: Utilize splay suitably.
        node = self.root
        while node is not None:
            if node.key == key:
                self.splay(node)
                return node
            if key < node.key:
                if node.left is None:
                    self.splay(node)
                    return None
                node = node.left
            else:
                if node.right is None:
                    self.splay(node)
                    return None
                node = node.right
        return None

    def insert(self, key):
        """Insert key into the tree.

        If the key is already present, nothing happens.
        """
        # TODO: Utilize splay suitably.
        if self.root is None:
            self.root = Node(key)
            return

        node = self.root
        while node.key != key:
            if key < node.key:
                if node.left is None:
                    node.left = Node(key, parent=node)
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(key, parent=node)
                node = node.right
        self.splay(node)

    def find_min(self, node):
        while node.left is not None:
            node = node.left
        return node

    def remove(self, key):
        """Remove given key from the tree.

        It the key is not present, nothing happens.
        """
        # TODO: Utilize splay suitably.
        node = self.root

        # 1. Find the node and splay it
        while node is not None and node.key != key:
            if key < node.key:
                if node.left is None:
                    self.splay(node)
                    return
                node = node.left
            else:
                if node.right is None:
                    self.splay(node)
                    return
                node = node.right

        if node is not None:
            self.splay(node)
            # 2.1 We remove the root, which splits the tree to a left subtree L and a right subtree R.
            left = node.left
            right = node.right
            if left is not None:
                left.parent = None

            # 2.2 if R is empty, we stop
            if right is None:
                self.root = left
                return
            else:
                right.parent = None

            # 3. We fint the minimum m of R and splay it. We note that m has no left child.
            r_min = self.find_min(node.right)
            self.splay(r_min)

            # 4. We connect the root of L as the left child of m.
            r_min.left = left
            left.parent = r_min

    def can_zig_zig(self, node):
        np = node.parent
        if np.left == node: # node is left child
            if np.parent.left == np: # parent is left child
                return True
        else: # node is right child
            if np.parent.right == np:
                return True
        return False

    def splay(self, node):
        """Splay the given node.

        If a single rotation needs to be performed, perform it as the last rotation
        (i.e., to move the splayed node to the root of the tree).
        """
        # TODO: Implement
        while node.parent is not None:
            if node.parent.parent is None: 
                # do single zig step
                self.rotate(node)
            else: 
                # can do double steps
                if self.can_zig_zig(node):
                    # zig-zig step
                    self.rotate(node.parent)
                    self.rotate(node)
                else:
                    # zig-zag step
                    self.rotate(node)
                    self.rotate(node)
