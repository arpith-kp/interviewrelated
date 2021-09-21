from collections import defaultdict
from typing import List, Optional

from src.leetcode.treeutil import TreeNode, createTree


def shortest_path():
    def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int, used=None, res=None) -> int:
        if used is None:
            used = {}
        if res is None:
            res = []
        if k+1 == len(res) and src in used:
            return sum(res)
        for trip in flights:
            cur_src, cur_dst, cost = trip
            if cur_src in used: continue
            used[cur_src] = True
            res.append(cost)
            result = findCheapestPrice(n, flights, src, dst, k, used, res)
            if result > 0: return result
            used[cur_src] = False
            res.pop()
        return 0

    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    src = 0
    dst = 2
    k = 0
    print(findCheapestPrice(n,flights,src,dst,k))

def leaves():
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

def max_diff_ancestor():
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

def concat_word():
    def findAllConcatenatedWordsInADict( words):
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

concat_word()
