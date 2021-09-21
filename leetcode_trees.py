import collections
from collections import OrderedDict
from typing import List, Optional

from src.leetcode.treeutil import TreeNode, Node
from src.leetcode.treeutil import createTree


class TreeObj:

    def __init__(self, left=None, right=None, next=None):
        left = left
        right = right
        data = None
        next_ptr = next


def countGoodNodes():
    """
    Adding globally from left and right, instead of passing varirable compute locally . res in below
    :return:
    """

    def cn(treenode: TreeObj, max_cur=None, good_nodes=0, level=1):
        if treenode is None:
            return 0
        if max_cur is None:
            max_cur = treenode.data

        res = 1 if treenode.data >= max_cur else 0

        max_cur = max(max_cur, treenode.data)
        print(" " * level, f" Max cur {max_cur} good nodes {res} currentNod {treenode.data}")
        res += cn(treenode.left, max_cur, good_nodes, level * 2)
        res += cn(treenode.right, max_cur, good_nodes, level * 2)
        return res

    root = TreeObj()
    root.data = 3
    root.left = TreeObj()
    root.left.data = 1
    root.right = TreeObj()
    root.right.data = 4
    root.left.left = TreeObj()
    root.left.left.data = 3
    root.right.left = TreeObj()
    root.right.left.data = 1
    root.right.right = TreeObj()
    root.right.right.data = 5
    gn = 0
    print(cn(root, good_nodes=gn))


def wordladder():
    def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:

        def wl(beginWord, endWord, wordList: List, level=1):
            print(" " * level, f" begin {beginWord} result {result}")
            ret = False
            if beginWord == endWord:
                return True
            for w in wordList:
                diff = len(set(beginWord) - set(w))
                if diff == 1 and w not in result:
                    result.append(w)
                    beginWord = w
                    if wl(beginWord, endWord, wordList, level * 2):
                        ret = True
                        break
            return ret

        result = []
        result.append(beginWord)
        result.append(endWord)
        wl(beginWord, endWord, wordList)
        if endWord in wordList:
            wl(beginWord, endWord, wordList)
        return len(result) - 2 if endWord in wordList and len(result) > 0 else 0

    print(ladderLength('hot', 'dog', ['hot', 'dog', "dot"]))


class Sol:
    """
    Final solution for Trees
    """

    def __init__(self):
        deapth = 0
        sums = 0

    def deepestLeavesSum(self):

        def deepestLeavesSum(root: Optional[TreeNode], lvl=0) -> int:
            if root is None or (root.val is None and root.right is None and root.left is None):
                return 0
            if lvl == self.deapth:
                self.sums += root.val
            elif lvl > self.deapth:
                sums = root.val
                deapth = lvl
            deepestLeavesSum(root.left, lvl + 1)
            deepestLeavesSum(root.right, lvl + 1)

        a = [1, 2, 3, 4, 5, None, 6, 7, None, None, None, None, None]
        root = None
        root = createTree(a, root)
        deepestLeavesSum(root)
        print(self.sums)
        return self.sums

    def sol2(self):
        """
        Breadth first search
        :return:
        """

        def deepestLeavesSum(root: TreeNode) -> int:

            q, ans, qlen, curr = [root], 0, 0, 0
            while len(q):
                qlen, ans = len(q), 0
                for _ in range(qlen):
                    curr = q.pop(0)
                    if curr is not None and curr.val is not None:
                        ans += curr.val
                    if curr.left: q.append(curr.left)
                    if curr.right: q.append(curr.right)
            return ans

        a = [1, 2, 3, 4, 5, None, 6, 7, None, None, None, None, 8]
        root = None
        root = createTree(a, root)
        print(deepestLeavesSum(root))


class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        message_store = OrderedDict()

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        if not message:
            return False

        if message in self.message_store:
            last_timestamp = self.message_store[message]
            if last_timestamp + 10 > timestamp:
                return False
        self.message_store[message] = timestamp
        return True


def emp():
    def getImportance(employees: List[int], id: int) -> int:
        """
        mistakes : should have used dict to store all employee so we can avoid loop and lookup by 1
        """
        res = 0
        check = []
        for emp in employees:
            cur_id, impo, subo = emp
            if cur_id == id:
                check.append(cur_id)

        while len(check) > 0:
            id = check[0]
            for emp in employees:

                cur_id, impo, subo = emp
                if cur_id == id:
                    res += impo
                    check.pop()

                    for s in subo:
                        check.append(s)

                    break
        return res

    employees = [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]]
    id = 1
    print(getImportance(employees, id))


def lca():
    def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

        if p.val < root.val and q.val < root.val:
            return lowestCommonAncestor(root.left, p, q)
        elif q.val > root.val and p.val > root.val:
            return lowestCommonAncestor(root.right, p, q)
        else:
            return root

    # a = [6, 2, 8, 0, 4, 7, 9, None, None, 3, 5]
    # p = TreeNode(data=3)
    # q = TreeNode(data=5)
    # root = None
    # root = createTree(a, root)
    # print(lowestCommonAncestor(root, p, q))

    def lowestCommonAncestors(root, p, q):
        while (root.val - p.val) * (root.val - q.val) > 0:
            root = (root.left, root.right)[p.val > root.val]
        return root

    def lca_binary_tree(root: TreeNode, p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
        if root is None or root.val == p.val or root.val == q.val:
            return root

        left = lca_binary_tree(root.left, p, q)
        right = lca_binary_tree(root.right, p, q)

        if left is not None and right is not None:
            return root
        return left if left is not None else right

    a = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    p = TreeNode(data=6)
    q = TreeNode(data=2)

    def maxAncestorDiff(root, mn=100000, mx=0):
        return max(maxAncestorDiff(root.left, min(mn, root.val), max(mx, root.val)), \
                   maxAncestorDiff(root.right, min(mn, root.val), max(mx, root.val))) \
            if root and root.val is not None else mx - mn

    a = [8, 3, 10, 1, 6, None, 14, None, None, 4, 7, 13]
    root = None

    def sumRootToLeaf(root: Optional[TreeNode], val=0) -> int:
        if not root: return 0
        val = val * 2 + root.val
        if root.left == root.right: return val
        return sumRootToLeaf(root.left, val) + sumRootToLeaf(root.right, val)

    a = [1, 0, 1, 0, 1, 0, 1]
    root = None
    root = createTree(a, root)
    print(sumRootToLeaf(root))


class Codec:

    def serialize(self, root):
        def preorder(node):
            if node:
                vals.append(str(node.val))
                preorder(node.left)
                preorder(node.right)

        vals = []
        preorder(root)
        return ' '.join(vals)

    def deserialize(self, data):
        preorder = map(int, data.split())
        inorder = sorted(preorder)
        return self.buildTree(preorder, inorder)

    def buildTree(self, preorder, inorder):
        def build(stop):
            if inorder and inorder[-1] != stop:
                root = TreeNode(preorder.pop())
                root.left = build(root.val)
                inorder.pop()
                root.right = build(stop)
                return root

        preorder.reverse()
        inorder.reverse()
        return build(None)

    def serialize2(self, root):
        """
        Encodes a tree to a single string.
        """

        def postorder(root):
            return postorder(root.left) + postorder(root.right) + [root.val] if root else []

        return ' '.join(map(str, postorder(root)))

    def deserialize2(self, data):
        """
        Decodes your encoded data to tree.
        """

        def helper(lower=float('-inf'), upper=float('inf')):
            if not data or data[-1] < lower or data[-1] > upper:
                return None

            val = data.pop()
            root = TreeNode(val)
            root.right = helper(val, upper)
            root.left = helper(lower, val)
            return root

        data = [int(x) for x in data.split(' ') if x]
        return helper()


class Solution2(object):
    def minDiffInBST(self, root):
        def dfs(node):
            if node:
                dfs(node.left)
                ans = min(self.ans, node.val - self.prev)
                prev = node.val
                dfs(node.right)

        prev = float('-inf')
        ans = float('inf')
        dfs(root)
        return ans


def check_complete():
    def isCompleteTree(root: Optional[TreeNode], level=0) -> bool:
        bfs = [root]
        i = 0
        while bfs[i] and bfs[i].val is not None:
            bfs.append(bfs[i].left)
            bfs.append(bfs[i].right)
            i += 1
        return not any(bfs[i:])

    # print(isCompleteTree(root))


def get_depath():
    def maxDepth(root: Optional[TreeNode]) -> int:
        bfs = [root]
        i = 0
        while len(bfs) > 0:
            size = len(bfs)
            while size > 0:
                a = bfs[0]
                bfs.pop(0)
                if a.left is not None:
                    bfs.append(a.left)
                if a.right is not None:
                    bfs.append(a.right)
                size -= 1
            i += 1
        return i

    a = [3, 9, 20, None, None, 15, 7]
    root = None
    root = createTree(a, root)
    maxDepth(root)


def any_sum():
    def anyS(node: Optional[TreeNode]) -> float:
        max_path = float("-inf")  # placeholder to be updated

        def get_max_gain(node):
            nonlocal max_path  # This tells that max_path is not a local variable
            if node is None:
                return 0
            gain_on_left = max(get_max_gain(node.left), 0)  # Read the part important observations
            gain_on_right = max(get_max_gain(node.right), 0)  # Read the part important observations
            current_max_path = node.val + gain_on_left + gain_on_right  # Read first three images of going down the recursion stack
            max_path = max(max_path, current_max_path)  # Read first three images of going down the recursion stack
            return node.val + max(gain_on_left, gain_on_right)  # Read the last image of going down the recursion stack

        get_max_gain(node)  # Starts the recursion chain
        return max_path

    a = [20, 15, 7]
    root = None
    root = createTree(a, root)
    print(anyS(root))


def topological_sort():
    graph = collections.defaultdict(list)
    visited = []
    res = []

    def findOrder(numCourses, prerequisites):
        # use DFS to parse the course structure
        # a graph for all courses
        nonlocal visited
        for pair in prerequisites:
            graph[pair[0]].append(pair[1])
        visited = [0 for x in range(numCourses)]  # DAG detection
        for x in range(numCourses):
            if not DFS(x):
                return []
            # continue to search the whole graph
        return res

    def DFS(node):
        if visited[node] == -1:  # cycle detected
            return False
        if visited[node] == 1:
            return True  # has been finished, and been added to res
        visited[node] = -1  # mark as visited
        for x in graph[node]:
            if not DFS(x):
                return False
        visited[node] = 1  # mark as finished
        res.append(node)  # add to solution as the course depenedent on previous ones
        return True

    numCourses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print(findOrder(numCourses, prerequisites))


def lca_with_parent():
    def lowestCommonAncestor(p: Node, q: Node) -> Node:
        p1, p2 = p, q
        while p1 != p2:
            p1 = p1.parent if p1.parent else q
            p2 = p2.parent if p2.parent else p

        return p1

    root = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    p = Node(5)
    q = Node(1)
    lowestCommonAncestor(p, q)


def topo_():
    def find_order(n, list_of_course: List[List[int]]) -> List:
        course = collections.defaultdict(list)
        visited = [0 for _ in range(n)]

        def dfs(cou, visited):
            if visited[cou] == -1:
                return False
            if visited[cou] == 1:
                return True
            visited[cou] = -1
            for k in course[cou]:
                if not dfs(k, visited):
                    return False
            res.append(cou)
            visited[cou] = 1
            return True

        res = []
        for c in list_of_course:
            from_c = c[0]
            to_c = c[1]
            course[from_c].append(to_c)
        for cou in range(n):
            if not dfs(cou, visited):
                return []
        return res

    n = 4
    prerequisites = [[1, 0], [0, 1], [3, 1], [3, 2]]
    print(find_order(n, prerequisites))


def nearest_neighbor():
    def findNearestRightNode(root: TreeNode, u: TreeNode) -> Optional[TreeNode]:
        next_level = collections.deque([root])

        while next_level:
            cur_level = next_level
            next_level = collections.deque()

            while cur_level:
                node = cur_level.popleft()
                if node and node.left:
                    next_level.append(node.left)
                if node and node.right:
                    next_level.append(node.right)

    a = [1, 2, 3, None, 4, 5, 6]
    u = TreeNode(data=4)
    root = None
    root = createTree(a, root)
    print(findNearestRightNode(root, u))


def hieght_subtree():
    def lcaDeepestLeaves(root):
        lca, deepest = None, 0

        def helper(node, depth):
            nonlocal lca, deepest
            deepest = max(deepest, depth)
            if not node:
                return depth
            left = helper(node.left, depth + 1)
            right = helper(node.right, depth + 1)
            if left == right == deepest:
                lca = node
            return max(left, right)

        helper(root, 0)
        return lca

    a = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    root = None
    root = createTree(a, root)
    print(lcaDeepestLeaves(root))


def sum_at_leve():
    def maxLevelSum(root: Optional[TreeNode]) -> int:
        queue = collections.deque([root, None])
        max_sum = float('-inf')
        level = 1
        max_level = 1
        sums = 0
        while queue:
            node = queue.popleft()
            if node and node.val is not None:
                sums += node.val
                if node.left and node.left.val is not None:
                    queue.append(node.left)
                if node.right and node.right.val is not None:
                    queue.append(node.right)
            else:
                #
                if sums > max_sum:
                    max_sum = sums
                    max_level = level
                if queue:
                    queue.append(None)
                    level += 1
                    sums = 0

        return max_level

    a = [1, 7, 0, 7, -8, None, None]
    root = None
    root = createTree(a, root)
    print(maxLevelSum(root))


def next_node_k_distant():
    def convert_into_graph(node, parent, g):
        # To convert into graph we need to know who is the parent
        if not node:
            return

        if parent:
            g[node].append(parent)

        if node.right:
            g[node].append(node.right)
            convert_into_graph(node.right, node, g)

        if node.left:
            g[node].append(node.left)
            convert_into_graph(node.left, node, g)

    def distanceK(root: TreeNode, target: TreeNode, K: int) -> List[int]:
        g = collections.defaultdict(list)
        vis, q, res = set(), collections.deque(), []
        # We have a graph, now we can use simply BFS to calculate K distance from node.
        convert_into_graph(root, None, g)

        q.append((target, 0))

        while q:
            n, d = q.popleft()
            vis.add(n)

            if d == K:
                res.append(n.val)

            # adjacency list traversal
            for nei in g[n]:
                if nei not in vis:
                    q.append((nei, d + 1))

        return res

    a = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    target = TreeNode(data=5)
    k = 2
    root = None
    root = createTree(a, root)
    distanceK(root, target, k)


def deapth_of_graph():
    def treeDiameter(edges: List[List[int]], move: int = 0) -> int:
        graph = collections.defaultdict(set)
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        bfs = {(u, None) for u, nex in graph.items() if len(nex) == 1}
        while bfs:
            bfs, move = {(v, u) for u, pre in bfs for v in graph[u] if v != pre}, move + 1
        return max(move - 1, 0)

    edges = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5]]
    print(treeDiameter(edges))


def duplicate_subtree():
    def findDuplicateSubtrees(root, heights=[]):
        def getid(root):
            if root:
                id = treeid[root.val, getid(root.left), getid(root.right)]
                trees[id].append(root)
                return id

        trees = collections.defaultdict(list)
        treeid = collections.defaultdict()
        treeid.default_factory = treeid.__len__
        getid(root)
        return [roots[0] for roots in trees.values() if roots[1:]]

    a = [1, 2, 3, 4, None, 2, 4, None, None, 4]
    root = None
    root = createTree(a, root)
    print(findDuplicateSubtrees(root))
duplicate_subtree()

def rob_tree():
    def rob(root: Optional[TreeNode]) -> int:
        def superrob(node):
            # returns tuple of size two (now, later)
            # now: max money earned if input node is robbed
            # later: max money earned if input node is not robbed

            # base case
            if not node: return (0, 0)

            # get values
            left, right = superrob(node.left), superrob(node.right)

            # rob now
            now = node.val + left[1] + right[1]

            # rob later
            later = max(left) + max(right)

            return (now, later)

        return max(superrob(root))

    a = [1, 2, 3, 4, None, 2, 4, None, None, 4]
    root = None
    root = createTree(a, root)
    print(rob(root))


def evenGrandparent():
    def sumEvenGrandparent(root: TreeNode) -> int:
        def dfs(root: TreeNode, parent: int = None, grandParent: int = None):
            if root is None:
                return 0

            return dfs(root.left, parent=root.val, grandParent=parent) + dfs(root.right, parent=root.val,
                                                                             grandParent=parent) + (
                       root.val if grandParent % 2 == 0 else 0)

        return dfs(root, 1, 1)

    a = [2, 3, 4, 4, 4, 4, 4]
    # a = [6,7,8,2,7,1,3,9,None,1,4,None,None,None,5]
    root = None
    root = createTree(a, root)
    print(sumEvenGrandparent(root))


def max_avg():
    def maximumAverageSubtree(root: Optional[TreeNode]) -> float:
        max_avg = 0.0

        def dfs(root: Optional[TreeNode]) -> tuple:
            nonlocal max_avg
            if root is None:
                return (0, root)

            left = dfs(root.left)
            right = dfs(root.right)

            left_c = left[0]
            right_c = right[0]

            avg = ((left[1].val if left[1] and left[1].val is not None else 0) + (
                right[1].val if right[1] and right[1].val is not None else 0) + (
                       root.val if root.val is not None else 0)) / float((left_c + right_c) + 1)
            max_avg = max(avg, max_avg)
            return ((left[0] + right[0] + 1), root)

        dfs(root)
        return max_avg

    a = [2, 6, 3, None, 5, 12, 7, None, None, 0, 4, None, 10, None, None, 11, None, None, 9, None, 1, None, 8]
    root = None
    root = createTree(a, root)
    print(maximumAverageSubtree(root))


def large_in_each_row():
    def largestValues(root: Optional[TreeNode]) -> List[int]:

        queue = collections.deque([root, None])
        tracker = []
        res = []
        while queue:

            node = queue.popleft()
            if node:

                if node.left and node.left is not None:
                    queue.append(node.left)
                if node.right and node.right is not None:
                    queue.append(node.right)
                if node.val is not None:
                    tracker.append(node.val)

            else:
                res.append(max(tracker))
                tracker = []
                if queue:
                    queue.append(None)
        return res

    a = [1, 3, 2, 5, 3, None, 9]
    root = None
    root = createTree(a, root)
    print(largestValues(root))


def deep_level():
    def subtreeWithAllDeepest(root):
        def deep(root):
            if not root: return 0, None
            l, r = deep(root.left), deep(root.right)
            if l[0] > r[0]:
                return l[0] + 1, l[1]
            elif l[0] < r[0]:
                return r[0] + 1, r[1]
            else:
                return l[0] + 1, root

        return deep(root)[1]

    a = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    root = None
    root = createTree(a, root)
    print(subtreeWithAllDeepest(root))


def trees_without_recursion():
    def connect(root):
        while root and root.left:
            next = root.left
            while root:
                root.left.next = root.right
                root.right.next = root.next and root.next.left
                root = root.next
            root = next

    a = [1, 2, 3, 4, 5, 6, 7]
    root = None
    root = createTree(a, root)
    print(connect(root))


def check_bst():
    def isValidBST(root: Optional[TreeNode]) -> bool:

        def dfs(root, left=float('-infinity'), right=float('infinity')):

            if root is None:
                return True

            if not left <= root.val <= right:
                return False

            return dfs(root.left, left, root.val) and dfs(root.right, root.val, right)

        print(dfs(root))

    a = [2, 2, 2]
    root = None
    root = createTree(a, root)
    isValidBST(root)


def largest_unique_val():
    def longestUnivaluePath(root: Optional[TreeNode]) -> int:
        longest = 0

        def dfs(node):
            nonlocal longest
            if node is None:
                return 0
            left_len = dfs(node.left)
            right_len = dfs(node.right)

            left = (left_len + 1) if node.left and node.left.val == node.val else 0
            right = (right_len + 1) if node.right and node.right.val == node.val else 0
            longest = max(longest, left + right)
            return max(left, right)

        dfs(root)
        return longest

    a = [1, 4, 5, 4, 4, 5]
    root = None
    root = createTree(a, root)
    print(longestUnivaluePath(root))


def large_bst():
    def largestBSTSubtree(root):
        def dfs(root):
            if not root:
                return 0, 0, float('inf'), float('-inf')
            N1, n1, min1, max1 = dfs(root.left)
            N2, n2, min2, max2 = dfs(root.right)
            n = n1 + 1 + n2 if max1 < root.val < min2 else float('-inf')
            return max(N1, N2, n), n, min(min1, root.val), max(max2, root.val)

        return dfs(root)[0]


def close_leaf():
    def findClosestLeaf(root: Optional[TreeNode], k: int) -> int:

        queue = collections.deque([root, None])
        node_found = False
        while queue:

            node = queue.popleft()
            if node and not node_found and node.val == k:
                node_found = True

            if node:
                if node_found and (node.left is None or node.left.val is None ) and (node.right is None or node.right.val is None) :
                    return node.val
                if node.left and node.left.val is not None :
                    queue.append(node.left)
                if node.right and node.right.val is not None:
                    queue.append(node.right)
            else:
                if queue:
                    queue.append(None)
        return None

    a = [1,2,3,None,None,4,5,6,None,None,7,8,9,10]

    k = 1
    root = None
    root = createTree(a, root)
    print(findClosestLeaf(root, k))

close_leaf()
