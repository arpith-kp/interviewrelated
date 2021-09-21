from collections import deque


# Node class for holding the Binary Tree
class TreeNode:
    def __init__(self, data=None):
        self.val = data
        self.left = None
        self.right = None
        self.next = None
    def __str__(self):
        return (f" {self.val}")


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
Q = deque()


# Helper function helps us in adding data
# to the tree in Level Order
def insertValue(data, root):

    newnode = TreeNode(data)
    if Q:
        temp = Q[0]
    if root == None:
        root = newnode

    # The left child of the current Node is
    # used if it is available.
    elif temp.left == None:
        temp.left = newnode

    # The right child of the current Node is used
    # if it is available. Since the left child of this
    # node has already been used, the Node is popped
    # from the queue after using its right child.
    elif temp.right == None:
        temp.right = newnode
        atemp = Q.popleft()

    # Whenever a new Node is added to the tree,
    # its address is pushed into the queue.
    # So that its children Nodes can be used later.
    Q.append(newnode)
    return root


# Function which calls add which is responsible
# for adding elements one by one
def createTree(a, root):
    for i in range(len(a)):
        root = insertValue(a[i], root)
    return root


# Function for printing level order traversal
def levelOrder(root):
    Q = deque()
    Q.append(root)
    while Q:
        temp = Q.popleft()
        print(temp.val, end=' ')
        if temp.left != None:
            Q.append(temp.left)
        if temp.right != None:
            Q.append(temp.right)
