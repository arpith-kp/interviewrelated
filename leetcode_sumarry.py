import heapq
import math
import operator
from collections import defaultdict, deque
from copy import copy
from typing import List, Optional

from src.leetcode.treeutil import TreeNode, createTree


def topological_sort():
    def find_order(n, list_of_course:List[List[int]])->List:
        course = defaultdict(list)
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
            visited[cou] =1
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

    n =4
    prerequisites = [[1, 0], [0, 1], [3, 1], [3, 2]]
    print(find_order(n, prerequisites))

def bfs():
    def bfs_level_order(root: TreeNode) -> None:
        next_level = deque([root])

        while next_level:
            cur_level = next_level
            next_level = deque()

            while cur_level:
                node = cur_level.popleft()
                if node and node.left:
                    next_level.append(node.left)
                if node and node.right:
                    next_level.append(node.right)


    def bfs_using_one_queue(root:TreeNode):
        """
        Initiate the queue by adding a root. Add None sentinel to mark the end of the first level.
        Initiate the current node as root.
        While queue is not empty:
        Pop the current node from the queue curr = queue.poll().
        If this node is u, return the next node from the queue. If there is no more nodes in the queue, return None.
        If the current node is not None:
        Add first left and then right child node into the queue.
        Update the current node: curr = queue.poll().
        Now, the current node is None, i.e. we reached the end of the current level. If the queue is not empty, push the None node as a sentinel, to mark the end of the next level.

        •	Use deque with None marker to traverse level order
        """
        if root is None:
            return None

        queue = deque([root, None, ])
        while queue:
            curr = queue.popleft()

            if curr:
                # add child nodes in the queue
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            else:
                # once the level is finished,
                # add a sentinel to mark end of level
                if queue:
                    queue.append(None)

    a = [1, 2, 3, None, 4, 5, 6]
    root = None
    root = createTree(a, root)
    bfs_level_order(root)



def hieght_subtree_bfs():
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


def sum_at_level_bfs():
    """perform something with bfs
    •    Do something with node at end of each level
    """
    def maxLevelSum(root: Optional[TreeNode]) -> int:
        queue = deque([root, None])
        max_sum = float('-inf')
        level = 1
        max_level = 1
        sums = 0
        while queue:
            node = queue.popleft()
            if node and node.val is not None:
                sums+=node.val
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
                    level+=1
                    sums=0

        return max_level

    a = [1, 7, 0, 7, -8, None, None]
    root = None
    root = createTree(a, root)
    print(maxLevelSum(root))

def lca():
    def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if root is None or root.val == p.val or root.val == q.val:
            return root

        left = lowestCommonAncestor(root.left, p, q)
        right = lowestCommonAncestor(root.right, p, q)

        if left is not None and right is not None:
            return root
        return left if left is not None else right

def grid_bfs():
    def closedIsland( grid: List[List[int]]) -> int:

        seen = set()
        def bfs(i, j):
            queue = deque([(i, j)])
            ans =1

            while queue:
                i, j = queue.popleft()
                seen.add((i,j))
                for r,c in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                        ans = 0
                    elif (r,c) not in seen and not grid[r][c]:
                        queue.append((r,c))
                        seen.add((r, c))
            return ans

        return sum(bfs(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if not cell and (i, j) not in seen)

    grid = [[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]
    print(closedIsland(grid))


def next_node_k_distant():
    """
    nodes are made as graph so can be traversed from any node to any.

    •	To traverse from any node to any node convert it to graph
    :return:
    """
    def convert_into_graph(node, parent, g):
        # To convert into graph we need to know who is the parent
        if not node:
            return

        if parent:
            g[node.val].append(parent)

        if node.right:
            g[node.val].append(node.right)
            convert_into_graph(node.right, node, g)

        if node.left:
            g[node.val].append(node.left)
            convert_into_graph(node.left, node, g)

    def distanceK(root: TreeNode, target: TreeNode, K: int) -> List[int]:
        g = defaultdict(list)
        vis, q, res = set(), deque(), []
        # We have a graph, now we can use simply BFS to calculate K distance from node.
        convert_into_graph(root, None, g)

        q.append((target.val, 0))

        while q:
            n, d = q.popleft()
            vis.add(n)

            if d == K and n is not None:
                res.append(n)

            # adjacency list traversal
            for nei in g[n]:
                if nei.val not in vis:
                    q.append((nei.val, d + 1))

        return res

    a = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    target = TreeNode(data=5)
    k = 2
    root = None
    root = createTree(a, root)
    print(distanceK(root,target, k))


def leaves_dfs():
    """
    get all leaves node. Store in default dict
    :return:
    """
    def findLeaves(root: TreeNode) -> List[List[int]]:
        output =defaultdict(list)

        def dfs(node, layer):
            if not node:
                return layer
            left = dfs(node.left, layer)
            right = dfs(node.right, layer)
            layer = max(left, right)
            output[layer].append(node.val)
            return layer + 1

        dfs(root, 0)
        return output.values()

    a = [1, 2, 3, 4, 5]
    root = None
    root = createTree(a, root)
    print(findLeaves(root))

def max_diff_ancestor_dfs():
    """
    preserve node as you iterate
    :return:
    """
    def maxAncestorDiff( root: Optional[TreeNode], mx=0, mn=100000) -> int:
        def dfs(node, high, low):
            if not node:
                return high - low
            high = max(high, node.val)
            low = min(low, node.val)
            return max(dfs(node.left, high, low), dfs(node.right, high, low))

        return dfs(root, root.val, root.val)


    a =  [1,2,3]
    # a =  [8,3,10,1,6,None,14,None,None,4,7,13]
    root = None
    root = createTree(a, root)
    print(maxAncestorDiff(root))


def combination3():
    """
    keep track of start and end. Keep changing target if solution found, clone stored result.
    :return:
    """
    def combinationSum(candidates, target):
        def backtrack(tmp, start, end, target):
            if target == 0:
                ans.append(tmp[:])
            elif target > 0:
                for i in range(start, end):
                    tmp.append(candidates[i])
                    backtrack(tmp, i, end, target - candidates[i])
                    tmp.pop()

        ans = []
        candidates.sort(reverse=True)
        backtrack([], 0, len(candidates), target)
        return ans

    candidates = [2, 3, 6, 7]
    target = 7
    print(combinationSum(candidates,target))


def general_matrix_solver():

    def helper(mat, a, b):
        """
        I use the bound to "shrink" the search space because once we hit a zero, there is no point in iterating past that point.
        O(m^2 x n^2)
        """
        m = len(mat)
        n = len(mat[0])

        count = 0
        bound = n

        for i in range(a,m):
            for j in range(b, bound):
                if mat[i][j]:
                    count+=1
                else:
                    bound=j
        return count

    def original_solver(mat):
        m = len(mat)
        n = len(mat[0])

        count = 0
        for i in range(m):
            for j in range(n):
                count+=helper(mat,i,j)

        return count

def substring_k_distinct():
    """
    Sliding window concepts
    :return:
    """
    def lengthOfLongestSubstringKDistinct(s, k):
        d = defaultdict(int)
        i, res = 0, 0
        for j in range(len(s)):
            d[s[j]] += 1
            # while have more than k, we keep removing from i
            while len(d) > k:
                d[s[i]] -= 1
                if d[s[i]] == 0:
                    del d[s[i]]
                i += 1
            res = max(res, j - i + 1)
        return res

    print(lengthOfLongestSubstringKDistinct('abccce', 3))

def sliding_without_duplicate():
    """
    Instead of dictionary count, add to set and remove items till that duplicate
    :return:
    """
    def sliding(S, K):
        res, i = 0, 0
        cur = set()
        for j in range(len(S)):
            while S[j] in cur:
                cur.remove(S[i])
                i += 1
            cur.add(S[j])
            res += j - i + 1 >= K
        return res

    s = "havefunonleetcode"
    k = 5
    print(sliding(s, k))

def longone():
    def longestOnes(nums: List[int], k: int) -> int:

        count = 0
        j = 0
        temp = 0
        for idx, i in enumerate(nums):
            if i ==0:
                temp+=1
            while temp > k:
                if nums[j] == 0:
                    temp-=1
                j+=1
            count = max(idx - j+1, count)
        return count

    nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]
    k = 3
    print(longestOnes(nums, k))

def subarray_product():
    """
    Sliding window without deque
    :return:
    """
    def numSubarrayProductLessThanK( nums: List[int], k: int) -> int:
        if k ==0:
            return 0
        count = 0
        product = 1
        j =0
        res = []
        for idx, i in enumerate(nums):
            product *= i
            while j<=i and product >= k:
                product = product/nums[j]
                j+=1
            if nums[j:idx+1] not in res:
                res.append(nums[j:idx+1])
            count+= idx - j +1
        print(res, len(res))
        return count

    nums = [10,9,10,4,3,8,3,3,6,2,10,10,9,3]
    k = 19
    print(numSubarrayProductLessThanK(nums, k))

def permuation_string():
    """
    String permutation. If you store string A count count in hashmap and to see if permutation of which exists in B simply
    check if count of those array char values are equal.

    Anything with string permuations use array to store values of its count and do match
    :return:
    """
    def checkInclusion(s1: str, s2: str) -> bool:
        A = [ord(x) - ord('a') for x in s1]
        B = [ord(x) - ord('a') for x in s2]

        target = [0] * 26
        for x in A:
            target[x] += 1

        window = [0] * 26
        for i, x in enumerate(B):
            window[x] += 1
            if i >= len(A):
                window[B[i - len(A)]] -= 1
            if window == target:
                return True
        return False

    s1 = "ab"
    s2 = "eidbaooo"
    print(checkInclusion(s1, s2))

def heap_k_small():
    def kthSmallest( matrix: List[List[int]], k: int) -> int:

        # The size of the matrix
        N = len(matrix)

        # Preparing our min-heap
        minHeap = []
        element = None
        for r in range(min(k, N)):
            # We add triplets of information for each cell
            minHeap.append((matrix[r][0], r, 0))

        # Heapify our list
        heapq.heapify(minHeap)

        # Until we find k elements
        while k:

            # Extract-Min
            element, r, c = heapq.heappop(minHeap)

            # If we have any new elements in the current row, add them
            if c < N - 1:
                heapq.heappush(minHeap, (matrix[r][c + 1], r, c + 1))

            # Decrement k
            k -= 1

        return element

    matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    k = 8
    print(kthSmallest(matrix,k))


def max_events():
    """
    Scheduling using heapq
    :return:
    """
    def maxEvents( events):
        events = sorted(events)
        total_days = max(event[1] for event in events)
        min_heap = []
        day, cnt, event_id = 1, 0, 0
        while day <= total_days:
            # if no events are available to attend today, let time flies to the next available event.
            if event_id < len(events) and not min_heap:
                day = events[event_id][0]

            # all events starting from today are newly available. add them to the heap.
            while event_id < len(events) and events[event_id][0] <= day:
                heapq.heappush(min_heap, events[event_id][1])
                event_id += 1

            # if the event at heap top already ended, then discard it.
            while min_heap and min_heap[0] < day:
                heapq.heappop(min_heap)

            # attend the event that will end the earliest
            if min_heap:
                heapq.heappop(min_heap)
                cnt += 1
            elif event_id >= len(events):
                break  # no more events to attend. so stop early to save time.
            day += 1

        return cnt
    d =[[1,4],[4,4],[2,2],[3,4],[1,1]]
    maxEvents(d)


def concat_word():
    """
    Word break always based on prefix and suffix
    :return:
    """
    def findAllConcatenatedWordsInADict(words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        d = set(words)
        memo = {}

        def dfs(word):
            if word in memo:
                return memo[word]
            memo[word] = False
            for i in range(1, len(word)):
                prefix = word[:i]
                suffix = word[i:]
                if prefix in d and suffix in d:
                    memo[word] = True
                    break
                if prefix in d and dfs(suffix):
                    memo[word] = True
                    break
            return memo[word]

        res = []
        for word in words:
            if dfs(word):
                res.append(word)

        return res

    words = ["cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"]
    print(findAllConcatenatedWordsInADict(words))

def palindrome_sub():
    def countSubstrings( s: str) -> int:

        dp = [[0] * len(s) for _ in range(len(s))]

        for idx in range(len(s)-1):
            dp[idx][idx] = 1
        res = 0

        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                if s[i]==s[j] and ((j-i+1)<3 or dp[i+1][j-1]):
                    dp[i][j] = 1
                    res+=dp[i][j]
        return res

    print(countSubstrings('aaa'))


def longestStrChain():
    def longestStrChain(words: List[str]) -> int:
        dp = {}
        result = 1

        for word in sorted(words, key=len):
            dp[word] = 1

            for i in range(len(word)):
                prev = word[:i] + word[i + 1:]

                if prev in dp:
                    dp[word] = max(dp[prev] + 1, dp[word])
                    result = max(result, dp[word])

        return result

    words = ["a", "b", "ba", "bca", "bda", "bdca"]
    print(longestStrChain(words))

def word_search_2():
    """DFS Right approach"""
    def exist(board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])

        def dfs(board, i, j, word, res=[]):
            if ''.join(res) == word:
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
                return False

            if board[i][j] != '#':
                res.append(board[i][j])
                c = board[i][j]
                board[i][j] = '#'
                value = dfs(board, i + 1, j, word, res) or\
                        dfs(board, i, j + 1, word, res) or\
                        dfs(board, i - 1, j, word, res) or \
                        dfs(board, i, j - 1, word, res)

                res.pop(-1)
                board[i][j] = c
                return value

            return False

        for i in range(m):
            for j in range(n):
                if dfs(board, i, j, word):
                    return True
        return False

    board = [["a"]]
    word = "a"
    print(exist(board,word))

def palindrome_partition():

    def partition(s: str) -> List[List[str]]:

        def backtrack(start, end, tmp=[]):
            if start == end:
                ans.append(tmp[:])
            for i in range(start, end):
                cur = s[start:i + 1]
                if cur == cur[::-1]:
                    tmp.append(cur)
                    backtrack(i + 1, end, tmp)
                    tmp.pop()

        ans = []
        backtrack(0, len(s))
        return ans

    s = "aab"
    print(partition(s))

def duplicate_subtree():
    def findDuplicateSubtrees(root, heights=[]):
        def getid(root):
            if root:
                id = treeid[root.val, getid(root.left), getid(root.right)]
                trees[id].append(root)
                return id

        trees = defaultdict(list)
        treeid = defaultdict()
        treeid.default_factory = treeid.__len__
        getid(root)
        return [roots[0] for roots in trees.values() if roots[1:]]



    a = [1,2,3,None,None,4,5,6]
    root = None
    root = createTree(a, root)
    graph=defaultdict(list)
    print(findDuplicateSubtrees(root))

def traverse_all_direction():
    def convert_into_graph(node, parent, g):
        # To convert into graph we need to know who is the parent
        if not node:
            return

        if parent:
            g[node.val].append(parent.val)

        if node.right:
            g[node.val].append(node.right.val)
            convert_into_graph(node.right, node, g)

        if node.left:
            g[node.val].append(node.left.val)
            convert_into_graph(node.left, node, g)

    def distanceK( root: TreeNode, target: TreeNode, K: int) -> List[int]:
        g = defaultdict(list)
        vis, q, res = set(), deque(), []
        # We have a graph, now we can use simply BFS to calculate K distance from node.
        convert_into_graph(root, None, g)

        q.append((target.val, 0))

        while q:
            n, d = q.popleft()
            vis.add(n)

            if d == K and n is not None:
                res.append(n)

            # adjacency list traversal
            for nei in g[n]:
                if nei.val not in vis:
                    q.append((nei.val, d + 1))

        return res
