# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class ValueNode():
    def __init__(self, t_value, n_value):
        self.t = t_value
        self.n = n_value
        self.left = None
        self.right = None


class Solution(object):
    def build_value_tree(self, root):
        if root is None:
            return None
        p_root = ValueNode(None, None)
        left, right = None, None
        if root.left is not None:
            left = self.build_value_tree(root.left)
        if root.right is not None:
            right = self.build_value_tree(root.right)
        p_root.left = left
        p_root.right = right
        return p_root

    def get_t_value(self, p_node, p_value_node):
        if p_node is None:
            return 0
        if p_value_node.t is not None:
            return p_value_node.t
        n_left = self.get_n_value(p_node.left, p_value_node.left)
        n_right = self.get_n_value(p_node.right, p_value_node.right)
        p_value_node.t = n_left + n_right + p_node.val
        return p_value_node.t

    def get_n_value(self, p_node, p_value_node):
        if p_node is None:
            return 0
        if p_value_node.n is not None:
            return p_value_node.n
        n_left = self.get_n_value(p_node.left, p_value_node.left)
        n_right = self.get_n_value(p_node.right, p_value_node.right)
        t_left = self.get_t_value(p_node.left, p_value_node.left)
        t_right = self.get_t_value(p_node.right, p_value_node.right)
        p_value_node.n = max(n_left, t_left) + max(n_right, t_right)
        return p_value_node.n

    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        value_root = self.build_value_tree(root)
        return max(self.get_t_value(root, value_root), self.get_n_value(root, value_root))



