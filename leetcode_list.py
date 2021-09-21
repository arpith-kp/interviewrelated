from typing import Optional


class TreeObj:

    def __init__(self, left=None, right=None, next=None):
        self.left = left
        self.right = right
        self.data = None
        self.next_ptr = next

    def __str__(self):
        return f"  {self.data}\n\t{self.left}\t{self.right}  "

    def generate_from_list(self, list_of_val):
        root = TreeObj()
        root.data = list_of_val[0]
        itr = iter(list_of_val[1:])
        cur = root
        for left in itr:
            right = next(itr)
            if left is not None:
                cur.left = TreeObj()
                cur.left.data = left
            if right is not None:
                cur.right = TreeObj()
                cur.right.data = right


def pathSum():
    def path(treenode, sums, prev=0, level=1):
        if treenode is None:
            return False
        prev += treenode.data
        print(" " * level, f"Checking node {treenode.data}", f"Current sum is {prev}. Final sum: {sums}")
        if prev == sums:
            return True
        x = path(treenode.left, sums, prev, level * 2)
        y = path(treenode.right, sums, prev, level * 2)
        return x or y

    root = TreeObj()
    root.data = 5
    root.left = TreeObj()
    root.left.data = 6
    root.right = TreeObj()
    root.right.data = 8
    root.left.left = TreeObj()
    root.left.left.data = 7
    root.left.left.left = TreeObj()
    root.left.left.left.data = 7
    root.left.left.right = TreeObj()
    root.left.left.right.data = 8
    root.right.left = TreeObj()
    root.right.left.data = 13
    root.right.right = TreeObj()
    root.right.right.data = 4
    root.right.right.right = TreeObj()
    root.right.right.right.data = 1

    print(path(root, 22))


def consecutive():
    def cons(treenode, results=None, level=1, final=0):
        if treenode is None:
            print(f" Reached end {final}")
            return final
        if results and treenode.data > results[-1]:
            final += 1
        elif not results:
            final += 1
        print(" " * level, f"Current data is {treenode.data}", f"final {final}")

        return max(cons(treenode.left, results, level * 2, final), cons(treenode.right, results, level * 2, final))

    root = TreeObj()
    root.data = 1
    root.left = TreeObj()
    root.left.data = 6
    root.right = TreeObj()
    root.right.data = 3
    root.left.left = TreeObj()
    root.left.left.data = 7
    root.left.left.left = TreeObj()
    root.left.left.left.data = 7
    root.left.left.right = TreeObj()
    root.left.left.right.data = 8
    root.right.left = TreeObj()
    root.right.left.data = 13
    root.right.right = TreeObj()
    root.right.right.data = 4
    root.right.right.left = TreeObj()
    root.right.right.left.data = 5

    print(cons(root))


def binaryMaximumPathSum():

    """
    all possible subtree, select one of subtree either left of right choose if that
    needs to be added
    :return:
    """
    sums = 0

    def tree(treenode: Optional[TreeObj],sums, level=1):

        if treenode is None:
            return 0
        left, right  = tree(treenode.left, sums, level * 2), tree(treenode.right,sums, level * 2)
        sums = max(sums, treenode.data + max(left, right))
        print(" " * level, f" Evaluating node {treenode.data}, current sum is {sums}")

        print(" " * level, f"Final solution is {sums} for node {treenode.data}")
        return max(left, right) + treenode.data

    root = TreeObj()
    root.data = 10
    root.left = TreeObj()
    root.left.data = 9
    root.right = TreeObj()
    root.right.data = 20
    root.right.right = TreeObj()
    root.right.right.data = 7
    root.right.left = TreeObj()
    root.right.left.data = 15

    print(tree(root, sums))

def levelOrderTraversal():

    def find_height(treenode, height=0):
        if treenode is None:
            return height

        return max(find_height(treenode.left, height+1), find_height(treenode.right, height+1))

    def lev(treenode, level=0, space=1):
        if treenode is None:
            return
        print(" " * space, f" Current level {level} and data is {treenode.data}, solution is {sol}")
        if level == len(sol):
            sol.append(list())
        sol[level].append(treenode.data)
        lev(treenode.left, level+1, space*2)
        lev(treenode.right, level + 1, space * 2)


    root = TreeObj()
    root.data = 3
    root.left = TreeObj()
    root.left.data = 9
    root.right = TreeObj()
    root.right.data = 20
    root.right.right = TreeObj()
    root.right.right.data = 7
    root.right.left = TreeObj()
    root.right.left.data = 15
    sol = []
    # for i in range(find_height(root)):
    #     sol.append(list())
    lev(root)
    # print("Hiehgt" , find_height(root))
    print(f"Solution is {sol}")


def nextPointer():

    def nextPointer(treenode: Optional[TreeObj], level=0, space=1):
        if treenode is None:
            return None
        print(" " * space, f"current at {treenode.data}")
        nextPointer(treenode.left, level + 1, space * 2)
        nextPointer(treenode.right, level + 1, space * 2)
        if treenode.left is not None and treenode.right is not None:
            print(" " * space, f"attaching {treenode.left.val} next pointer {treenode.right.val}")
            treenode.left.next_ptr = treenode.right

    root = TreeObj()
    root.data = 1
    root.left = TreeObj()
    root.left.data = 2
    root.right = TreeObj()
    root.right.data = 3
    root.right.right = TreeObj()
    root.right.right.data = 7
    root.right.left = TreeObj()
    root.right.left.data = 6

    root.left.right = TreeObj()
    root.left.right.data = 5
    root.left.left = TreeObj()
    root.left.left.data = 4

    nextPointer(root)

nextPointer()
