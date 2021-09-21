import collections
import heapq
import math
from typing import List


def numMatch():
    def numMatchingSubseq(S: str, words: List[str]) -> int:
        waiting = collections.defaultdict(list)
        for it in map(iter, words):
            waiting[next(it)].append(it)
        for c in S:
            for it in waiting.pop(c, ()):
                waiting[next(it, None)].append(it)
        return len(waiting[None])

    s = "abcde"
    words = ["a", "bb", "acd", "ace"]
    print(numMatchingSubseq(s, words))


def banana():
    def minEatingSpeed(piles, H):
        """
        :type piles: List[int]
        :type H: int
        :rtype: int
        """
        l, r = 1, max(piles)
        while l < r:
            m = l + (r - l) // 2
            time = sum([math.ceil(i / m) for i in piles])
            if time > H:
                l = m + 1
            else:
                r = m
        return l

    piles = [3, 6, 7, 11]
    h = 8
    minEatingSpeed(piles, h)


def maxKScores():
    def maxScore(cardPoints: List[int], k: int) -> int:
        size = len(cardPoints) - k
        minSubArraySum = float('inf')
        j = curr = 0

        for i, v in enumerate(cardPoints):
            curr += v
            if i - j + 1 > size:
                curr -= cardPoints[j]
                j += 1
            if i - j + 1 == size:
                minSubArraySum = min(minSubArraySum, curr)

        return sum(cardPoints) - minSubArraySum

    cardPoints = [1, 2, 3, 4, 5, 6, 1]
    k = 3
    print(maxScore(cardPoints, k))


def cpu():
    def getOrder(tasks: List[List[int]]) -> List[int]:

        res = []
        tasks = sorted([(t[0], t[1], i) for i, t in enumerate(tasks)])
        i = 0
        h = []
        time = tasks[0][0]
        while len(res) < len(tasks):
            while (i < len(tasks)) and (tasks[i][0] <= time):
                heapq.heappush(h, (tasks[i][1], tasks[i][2]))  # (processing_time, original_index)
                i += 1
            if h:
                t_diff, original_index = heapq.heappop(h)
                time += t_diff
                res.append(original_index)
            elif i < len(tasks):
                time = tasks[i][0]
        return res

    tasks = [[1, 2], [2, 4], [3, 2], [4, 1]]
    print(getOrder(tasks))


def longestIncreasing():
    """
    concise solution dp
    :return:
    """

    def longestIncreasingPath(matrix):
        def dfs(i, j):
            if not dp[i][j]:
                val = matrix[i][j]
                dp[i][j] = 1 + max(
                    dfs(i - 1, j) if i and val > matrix[i - 1][j] else 0,
                    dfs(i + 1, j) if i < M - 1 and val > matrix[i + 1][j] else 0,
                    dfs(i, j - 1) if j and val > matrix[i][j - 1] else 0,
                    dfs(i, j + 1) if j < N - 1 and val > matrix[i][j + 1] else 0)
            return dp[i][j]

        if not matrix or not matrix[0]: return 0
        M, N = len(matrix), len(matrix[0])
        dp = [[0] * N for i in range(M)]
        return max(dfs(x, y) for x in range(M) for y in range(N))


def studentRec():
    """
    Brute force solution: Generic one
    :return:
    """
    class Solution:

        def __init__(self):
            self.count = 0
            self.M = 1000000007

        def checkRecord(self, n) -> int:
            self.count = 0
            self.gen("", n)
            return self.count

        def gen(self, s, n):
            if n == 0 and self.checkRecordValid(s):
                self.count = (self.count + 1) % self.M
            elif n > 0:
                self.gen(s + "A", n - 1)
                self.gen(s + "L", n - 1)
                self.gen(s + "P", n - 1)

        def checkRecordValid(self, s):

            c = collections.Counter(s)
            try:
                lates = s.index('LLL')
            except ValueError:
                lates = -1
            return len(s) > 0 and lates < 0 and c['A'] < 2

    print(Solution().checkRecord(2))


studentRec()
