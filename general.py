from typing import List


def combinations():

    def combine(self, n, k):
        res = []
        self.dfs(range(1, n + 1), k, 0, [], res)
        return res


    def dfs(self, nums, k, index, path, res):
        # if k < 0:  #backtracking
        # return
        if k == 0:
            res.append(path)
            return  # backtracking
        for i in range(index, len(nums)):
            self.dfs(nums, k - 1, i + 1, path + [nums[i]], res)

def permutation():
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
            # return # backtracking
        for i in range(len(nums)):
            self.dfs(nums[:i] + nums[i + 1:], path + [nums[i]], res)

def combination_sum():
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res

    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return
        for i in range(index, len(nums)):
            self.dfs(nums, target - nums[i], i, path + [nums[i]], res)


def dfs_generic():
    def depth_first_search_recursive(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for next in graph[start] - visited:
            depth_first_search_recursive(graph, next, visited)
        return visited

def binary_search_generic():
    def binary_search(array) -> int:
        def condition(value) -> bool:
            pass

        left, right = 0, len(array)
        while left < right:
            mid = left + (right - left) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        return left


class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        def feasible(speed) -> bool:
            # return sum(math.ceil(pile / speed) for pile in piles) <= H  # slower
            return sum((pile - 1) / speed + 1 for pile in piles) <= H  # faster

        left, right = 1, max(piles)
        while left < right:
            mid = left  + (right - left) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left
