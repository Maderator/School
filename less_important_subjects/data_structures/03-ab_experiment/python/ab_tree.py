#!/usr/bin/env python3

import math
from typing import get_type_hints

class ABNode:
    """Single node in an ABTree.

    Each node contains keys and children
    (with one more children than there are keys).
    We also store a pointer to node's parent (None for root).
    """
    def __init__(self, keys = None, children = None, parent = None):
        self.keys = keys if keys is not None else []
        self.children = children if children is not None else []
        self.parent = parent

    def find_branch(self, key):
        """ Try finding given key in this node.

        If this node contains the given key, returns (True, key_position).
        If not, returns (False, first_position_with_key_greater_than_the_given).
        """
        i = 0
        while (i < len(self.keys) and self.keys[i] < key):
            i += 1

        return (i < len(self.keys) and self.keys[i] == key, i)

    def insert_branch(self, i, key, child):
        """ Insert a new key and a given child between keys i and i+1."""
        self.keys.insert(i, key)
        self.children.insert(i + 1, child)
    
    def set_node_parent(self, node, parent):
        if node is not None:
            node.parent = parent

    def insert_split_children(self, i, key, left_child, right_child):
        """ Insert a splitted child node given by key, left_child, and right_child between keys i and i+1."""
        self.insert_branch(i, key, right_child)
        self.children.pop(i)
        self.children.insert(i, left_child)
        self.set_node_parent(left_child, self)
        self.set_node_parent(right_child, self)
class ABTree:
    """A class representing the whole ABTree."""
    def __init__(self, a, b):
        assert a >= 2 and b >= 2 * a - 1, "Invalid values of a, b: {}, {}".format(a, b)
        self.a = a
        self.b = b
        self.root = ABNode(children=[None])

    def find(self, key):
        """Find a key in the tree.

        Returns True if the key is present, False otherwise.
        """
        node = self.root
        while node:
            found, i = node.find_branch(key)
            if found: return True
            node = node.children[i]
        return False
    
    def set_children_parent(self, children, parent):
        """Helper function for split_node

        set parent of each children
        """
        for child in children:
            self.set_node_parent(child, parent)

    def set_node_parent(self, node, parent):
        """Helper function for insert

        set parent of node if node is not None
        """
        if node is not None:
            node.parent = parent

    def split_node(self, node, size):
        """Helper function for insert

        Split node into two nodes such that original node contains first _size_ children.
        Return new node and the key separating nodes.
        """
        # KEY SEPARATING NODES
        key = node.keys[size]
        
        # NEW NODE 
        new_node = ABNode(keys=node.keys[size+1:], children=node.children[size+1:])
        self.set_children_parent(new_node.children, new_node)
        
        # ORIGINAL NODE
        node.keys = node.keys[:size]
        node.children = node.children[:size+1]

        return new_node, key

    def insert(self, key):
        """Add a given key to the tree, unless already present."""
        t = self.insert2(self.root, key)
        if t is not None:
            (left_child, new_root_key, right_child) = t
            new_root = ABNode(keys=[new_root_key], children=[left_child, right_child])
            self.set_node_parent(left_child, new_root)
            self.set_node_parent(right_child, new_root)
            self.root = new_root

    def is_list(self, node):
        """Helper function for insert2

        Find out if the node is list (i.e. it is None or it has no keys)
        """
        if node is None or len(node.keys) == 0:
            return True
        else:
            return False

    def insert2(self, root, key):
        """Add a given key to the tree, unless already present. Returns left subtree, new root, and right subtree, or None"""
        if self.is_list(root):
            return (None, key, None)
        (has_key, pos) = root.find_branch(key)
        if has_key:
            return None
        t = self.insert2(root.children[pos], key)
        if t is None:
            return None
        (left_child, new_root_key, right_child) = t
        root.insert_split_children(pos, new_root_key, left_child, right_child)
        if len(root.children) <= self.b:
            return None
        m = math.floor((self.b-1) / 2) # index of middle key of b keys
        right_child, xm = self.split_node(root, m)
        return (root, xm, right_child)
        
        